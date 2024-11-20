import math
from collections import deque, OrderedDict
from typing import Dict, Any, List, Optional, Union

import numpy as np
import tree
from ray.rllib.connectors.common import AddStatesFromEpisodesToBatch
from ray.rllib.core import Columns, DEFAULT_MODULE_ID
from ray.rllib.core.rl_module import RLModule, MultiRLModule
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.postprocessing.zero_padding import create_mask_and_seq_lens
from ray.rllib.utils.spaces.space_utils import BatchedNdArray, batch
from ray.rllib.utils.typing import EpisodeType


class AddStatesFromEpisodesToBatchForDictSpace(AddStatesFromEpisodesToBatch):

    def __call__(self, *,
                 rl_module: RLModule,
                 batch: Dict[str, Any],
                 episodes: List[EpisodeType],
                 explore: Optional[bool] = None,
                 shared_data: Optional[dict] = None,
                 **kwargs):
        # If not stateful OR STATE_IN already in data, early out.
        if not rl_module.is_stateful() or Columns.STATE_IN in batch:
            return batch

        # Make all inputs (other than STATE_IN) have an additional T-axis.
        # Since data has not been batched yet (we are still operating on lists in the
        # batch), we add this time axis as 0 (not 1). When we batch, the batch axis will
        # be 0 and the time axis will be 1.
        # Also, let module-to-env pipeline know that we had added a single timestep
        # time rank to the data (to remove it again).
        if not self._as_learner_connector:
            for column in batch.keys():
                self.foreach_batch_item_change_in_place(
                    batch=batch,
                    column=column,
                    func=lambda item, eps_id, aid, mid: (
                        item
                        if mid is not None and not rl_module[mid].is_stateful()
                        # Expand on axis 0 (the to-be-time-dim) if item has not been
                        # batched yet, otherwise axis=1 (the time-dim).
                        else tree.map_structure(
                            lambda s: np.expand_dims(
                                s, axis=(1 if isinstance(s, BatchedNdArray) else 0)
                            ),
                            item,
                        )
                    ),
                )
            shared_data["_added_single_ts_time_rank"] = True
        else:
            # Before adding STATE_IN to the `data`, zero-pad existing data and batch
            # into max_seq_len chunks.
            for column, column_data in batch.copy().items():
                # Do not zero-pad INFOS column.
                if column == Columns.INFOS:
                    continue
                for key, item_list in column_data.items():
                    # Multi-agent case AND RLModule is not stateful -> Do not zero-pad
                    # for this model.
                    assert isinstance(key, tuple)
                    mid = None
                    if len(key) == 3:
                        eps_id, aid, mid = key
                        if not rl_module[mid].is_stateful():
                            continue
                    column_data[key] = split_and_zero_pad(
                        item_list,
                        max_seq_len=self._get_max_seq_len(rl_module, module_id=mid),
                    )
                    # TODO (sven): Remove this hint/hack once we are not relying on
                    #  SampleBatch anymore (which has to set its property
                    #  zero_padded=True when shuffling).
                    shared_data[
                        (
                            "_zero_padded_for_mid="
                            f"{mid if mid is not None else DEFAULT_MODULE_ID}"
                        )
                    ] = True

        for sa_episode in self.single_agent_episode_iterator(
                episodes,
                # If Learner connector, get all episodes (for train batch).
                # If EnvToModule, get only those ongoing episodes that just had their
                # agent step (b/c those are the ones we need to compute actions for next).
                agents_that_stepped_only=not self._as_learner_connector,
        ):
            if self._as_learner_connector:
                assert sa_episode.is_finalized

                # Multi-agent case: Extract correct single agent RLModule (to get the
                # state for individually).
                sa_module = rl_module
                if sa_episode.module_id is not None:
                    sa_module = rl_module[sa_episode.module_id]
                else:
                    sa_module = (
                        rl_module[DEFAULT_MODULE_ID]
                        if isinstance(rl_module, MultiRLModule)
                        else rl_module
                    )
                # This single-agent RLModule is NOT stateful -> Skip.
                if not sa_module.is_stateful():
                    continue

                max_seq_len = sa_module.model_config["max_seq_len"]

                # look_back_state.shape=([state-dim],)
                look_back_state = (
                    # Episode has a (reset) beginning -> Prepend initial
                    # state.
                    convert_to_numpy(sa_module.get_initial_state())
                    if sa_episode.t_started == 0
                    # Episode starts somewhere in the middle (is a cut
                    # continuation chunk) -> Use previous chunk's last
                    # STATE_OUT as initial state.
                    else sa_episode.get_extra_model_outputs(
                        key=Columns.STATE_OUT,
                        indices=-1,
                        neg_index_as_lookback=True,
                    )
                )
                # state_outs.shape=(T,[state-dim])  T=episode len
                state_outs = sa_episode.get_extra_model_outputs(key=Columns.STATE_OUT)
                self.add_n_batch_items(
                    batch=batch,
                    column=Columns.STATE_IN,
                    # items_to_add.shape=(B,[state-dim])
                    # B=episode len // max_seq_len
                    items_to_add=tree.map_structure(
                        # Explanation:
                        # [::max_seq_len]: only keep every Tth state.
                        # [:-1]: Shift state outs by one, ignore very last
                        # STATE_OUT (but therefore add the lookback/init state at
                        # the beginning).
                        lambda i, o, m=max_seq_len: np.concatenate([[i], o[:-1]])[::m],
                        look_back_state,
                        state_outs,
                    ),
                    num_items=int(math.ceil(len(sa_episode) / max_seq_len)),
                    single_agent_episode=sa_episode,
                )

                # Also, create the loss mask (b/c of our now possibly zero-padded data)
                # as well as the seq_lens array and add these to `data` as well.
                mask, seq_lens = create_mask_and_seq_lens(len(sa_episode), max_seq_len)
                self.add_n_batch_items(
                    batch=batch,
                    column=Columns.SEQ_LENS,
                    items_to_add=seq_lens,
                    num_items=len(seq_lens),
                    single_agent_episode=sa_episode,
                )
                if not shared_data.get("_added_loss_mask_for_valid_episode_ts"):
                    self.add_n_batch_items(
                        batch=batch,
                        column=Columns.LOSS_MASK,
                        items_to_add=mask,
                        num_items=len(mask),
                        single_agent_episode=sa_episode,
                    )
            else:
                assert not sa_episode.is_finalized

                # Multi-agent case: Extract correct single agent RLModule (to get the
                # state for individually).
                sa_module = rl_module
                if sa_episode.module_id is not None:
                    sa_module = rl_module[sa_episode.module_id]
                # This single-agent RLModule is NOT stateful -> Skip.
                if not sa_module.is_stateful():
                    continue

                # Episode just started -> Get initial state from our RLModule.
                if sa_episode.t_started == 0 and len(sa_episode) == 0:
                    state = sa_module.get_initial_state()
                # Episode is already ongoing -> Use most recent STATE_OUT.
                else:
                    state = sa_episode.get_extra_model_outputs(
                        key=Columns.STATE_OUT, indices=-1
                    )
                self.add_batch_item(
                    batch,
                    Columns.STATE_IN,
                    item_to_add=state,
                    single_agent_episode=sa_episode,
                )

        return batch


def split_and_zero_pad(
        item_list: List[Union[BatchedNdArray, np._typing.NDArray, float]],
        max_seq_len: int,
) -> List[np._typing.NDArray]:
    """Splits the contents of `item_list` into a new list of ndarrays and returns it.

    In the returned list, each item is one ndarray of len (axis=0) `max_seq_len`.
    The last item in the returned list may be (right) zero-padded, if necessary, to
    reach `max_seq_len`.

    If `item_list` contains one or more `BatchedNdArray` (instead of individual
    items), these will be split accordingly along their axis=0 to yield the returned
    structure described above.

    .. testcode::

        from ray.rllib.utils.postprocessing.zero_padding import (
            BatchedNdArray,
            split_and_zero_pad,
        )
        from ray.rllib.utils.test_utils import check

        # Simple case: `item_list` contains individual floats.
        check(
            split_and_zero_pad([0, 1, 2, 3, 4, 5, 6, 7], 5),
            [[0, 1, 2, 3, 4], [5, 6, 7, 0, 0]],
        )

        # `item_list` contains BatchedNdArray (ndarrays that explicitly declare they
        # have a batch axis=0).
        check(
            split_and_zero_pad([
                BatchedNdArray([0, 1]),
                BatchedNdArray([2, 3, 4, 5]),
                BatchedNdArray([6, 7, 8]),
            ], 5),
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 0]],
        )

    Args:
        item_list: A list of individual items or BatchedNdArrays to be split into
            `max_seq_len` long pieces (the last of which may be zero-padded).
        max_seq_len: The maximum length of each item in the returned list.

    Returns:
        A list of np.ndarrays (all of length `max_seq_len`), which contains the same
        data as `item_list`, but split into sub-chunks of size `max_seq_len`.
        The last item in the returned list may be zero-padded, if necessary.
    """
    zero_element = tree.map_structure(
        lambda s: np.zeros_like([s[0]] if isinstance(s, BatchedNdArray) else s),
        item_list[0],
    )

    # The replacement list (to be returned) for `items_list`.
    # Items list contains n individual items.
    # -> ret will contain m batched rows, where m == n // T and the last row
    # may be zero padded (until T).
    ret = []

    # List of the T-axis item, collected to form the next row.
    current_time_row = []
    current_t = 0

    item_list = deque(item_list)
    while len(item_list) > 0:
        item = item_list.popleft()
        # `item` is already a batched np.array: Split if necessary.
        if isinstance(item, BatchedNdArray):
            t = max_seq_len - current_t
            current_time_row.append(item[:t])
            if len(item) <= t:
                current_t += len(item)
            else:
                current_t += t
                item_list.appendleft(item[t:])
        elif isinstance(item, dict):
            tempdict = OrderedDict()
            tempdict2 = OrderedDict()
            t = max_seq_len - current_t
            tempVal = None
            for key, val in item.items():
                tempVal = val
                if isinstance(val, BatchedNdArray):
                    tempdict[key] = val[:t]
                if not (len(val) <= t):
                    tempdict2[key] = val[t:]
            if len(tempVal) <= t:
                current_t += len(tempVal)
            else:
                current_t += t
                item_list.appendleft(tempdict2)
            current_time_row.append(tempdict)
        # `item` is a single item (no batch axis): Append and continue with next item.
        else:
            current_time_row.append(item)
            current_t += 1

        # `current_time_row` is "full" (max_seq_len): Append as ndarray (with batch
        # axis) to `ret`.
        if current_t == max_seq_len:
            ret.append(
                batch(
                    current_time_row,
                    individual_items_already_have_batch_dim="auto",
                )
            )
            current_time_row = []
            current_t = 0

    # `current_time_row` is unfinished: Pad, if necessary and append to `ret`.
    if current_t > 0 and current_t < max_seq_len:
        current_time_row.extend([zero_element] * (max_seq_len - current_t))
        ret.append(
            batch(current_time_row, individual_items_already_have_batch_dim="auto")
        )
    return ret
