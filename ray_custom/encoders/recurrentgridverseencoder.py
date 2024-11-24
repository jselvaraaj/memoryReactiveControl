from dataclasses import dataclass

import tree
from ray.rllib.core import Columns
from ray.rllib.core.models.base import ENCODER_OUT, Encoder
from ray.rllib.core.models.configs import RecurrentEncoderConfig
from ray.rllib.core.models.torch.encoder import TorchLSTMEncoder
from ray.rllib.policy.rnn_sequencing import get_fold_unfold_fns


@dataclass
class RecurrentGridverseEncoderConfig(RecurrentEncoderConfig):
    def build(self, framework: str = "torch"):
        if framework != "torch":
            raise Exception("RecurrentGridverseEncoder is only available in pytorch")
        if self.recurrent_layer_type != "lstm":
            raise Exception("RecurrentGridverseEncoder is only available for recurrent_layer_type of lstm")
        return TorchLSTMDictSpaceEncoder(self)


class TorchLSTMDictSpaceEncoder(TorchLSTMEncoder):
    def __init__(self, config: RecurrentEncoderConfig) -> None:
        super().__init__(config)
        if self.tokenizer is None:
            raise Exception("RecurrentGridverseEncoder only works with a tokenizer")

    def _forward(self, inputs: dict, **kwargs) -> dict:
        outputs = {}

        # Push observations through the tokenizer encoder if we built one.
        out = tokenize(self.tokenizer, inputs, framework="torch")

        # States are batch-first when coming in. Make them layers-first.
        # noinspection DuplicatedCode
        states_in = tree.map_structure(
            lambda s: s.transpose(0, 1), inputs[Columns.STATE_IN]
        )

        out, states_out = self.lstm(out, (states_in["h"], states_in["c"]))
        states_out = {"h": states_out[0], "c": states_out[1]}

        # Insert them into the output dict.
        outputs[ENCODER_OUT] = out
        outputs[Columns.STATE_OUT] = tree.map_structure(
            lambda s: s.transpose(0, 1), states_out
        )
        return outputs


# noinspection DuplicatedCode,PyIncorrectDocstring
def tokenize(tokenizer: Encoder, inputs: dict, framework: str) -> dict:
    """Tokenizes the observations from the input dict.

    Args:
        tokenizer: The tokenizer to use.
        inputs: The input dict.

    Returns:
        The output dict.
    """
    # Tokenizer may depend solely on observations.
    obs = inputs[Columns.OBS]

    size = list(list(obs.values())[0].size() if framework == "torch" else obs.shape)
    b_dim, t_dim = size[:2]
    fold, unfold = get_fold_unfold_fns(b_dim, t_dim, framework=framework)
    # Push through the tokenizer encoder.
    tokenizer_input = {Columns.OBS: fold(obs)}
    # noinspection PyCallingNonCallable
    out = tokenizer(tokenizer_input)
    out = out[ENCODER_OUT]
    # Then unfold batch- and time-dimensions again.
    return unfold(out)
