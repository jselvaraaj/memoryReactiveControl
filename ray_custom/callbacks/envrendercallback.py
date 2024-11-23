from typing import Optional, Sequence, Union, List

import gymnasium as gym
import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.images import resize
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType


# noinspection DuplicatedCode
class EnvRenderCallback(DefaultCallbacks):
    def __init__(self, env_runner_indices: Optional[Sequence[int]] = None):
        super().__init__()
        # Only render and record on certain EnvRunner indices?
        self.env_runner_indices = env_runner_indices
        # Per sample round (on this EnvRunner), we want to only log the best- and
        # worst performing episode's videos in the custom metrics. Otherwise, too much
        # data would be sent to WandB.
        self.best_episode_and_return = (None, float("-inf"))
        self.worst_episode_and_return = (None, float("inf"))

    def on_episode_step(
            self,
            *,
            episode: Union[EpisodeType, Episode, EpisodeV2],
            env_runner: Optional["EnvRunner"] = None,
            metrics_logger: Optional[MetricsLogger] = None,
            env: Optional[gym.Env] = None,
            env_index: int,
            rl_module: Optional[RLModule] = None,
            **kwargs,
    ) -> None:
        if not (isinstance(env_runner, SingleAgentEnvRunner) or isinstance(env_runner, MultiAgentEnvRunner)):
            raise Exception("env_runner is not SingleAgentEnvRunner or MultiAgentEnvRunner in EnvRenderCallback")
        if (
                self.env_runner_indices is not None
                and env_runner.worker_index not in self.env_runner_indices
        ):
            return

        # If we have a vector env, only render the sub-env at index 0.
        if isinstance(env.unwrapped, gym.vector.VectorEnv):
            image = env.envs[0].render()
        # Render the gym.Env.
        else:
            image = env.render()

        # Original render images for CartPole are 400x600 (hxw). We'll downsize here to
        # a very small dimension (to save space and bandwidth).
        image = resize(image, 64, 96)
        # For WandB videos, we need to put channels first.
        image = np.transpose(image, axes=[2, 0, 1])
        # Add the compiled single-step image as temp. data to our Episode object.
        # Once the episode is done, we'll compile the video from all logged images
        # and log the video with the EnvRunner's `MetricsLogger.log_...()` APIs.
        # See below:
        # `on_episode_end()`: We compile the video and maybe store it).
        # `on_sample_end()` We log the best and worst video to the `MetricsLogger`.
        episode.add_temporary_timestep_data("render_images", image)

    def on_episode_end(
            self,
            *,
            episode: Union[EpisodeType, Episode, EpisodeV2],
            env_runner: Optional["EnvRunner"] = None,
            metrics_logger: Optional[MetricsLogger] = None,
            env: Optional[gym.Env] = None,
            env_index: int,
            rl_module: Optional[RLModule] = None,
            **kwargs,
    ) -> None:
        # Get the episode's return.
        episode_return = episode.get_return()

        # Better than the best or worse than worst Episode thus far?
        if (
                episode_return > self.best_episode_and_return[1]
                or episode_return < self.worst_episode_and_return[1]
        ):
            # Pull all images from the temp. data of the episode.
            images = episode.get_temporary_timestep_data("render_images")
            # `images` is now a list of 3D ndarrays

            # Create a video from the images by simply stacking them AND
            # adding an extra B=1 dimension. Note that Tune's WandB logger currently
            # knows how to log the different data types by the following rules:
            # array is shape=3D -> An image (c, h, w).
            # array is shape=4D -> A batch of images (B, c, h, w).
            # array is shape=5D -> A video (1, L, c, h, w), where L is the length of the
            # video.
            # -> Make our video ndarray a 5D one.
            video = np.expand_dims(np.stack(images, axis=0), axis=0)

            # `video` is from the best episode in this cycle (iteration).
            if episode_return > self.best_episode_and_return[1]:
                self.best_episode_and_return = (video, episode_return)
            # `video` is worst in this cycle (iteration).
            else:
                self.worst_episode_and_return = (video, episode_return)

    def on_sample_end(
            self,
            *,
            env_runner: Optional["EnvRunner"] = None,
            metrics_logger: Optional[MetricsLogger] = None,
            samples: Union[SampleBatch, List[EpisodeType]],
            **kwargs,
    ) -> None:
        # Best video.
        if self.best_episode_and_return[0] is not None:
            metrics_logger.log_value(
                "episode_videos_best",
                self.best_episode_and_return[0],
                # Do not reduce the videos (across the various parallel EnvRunners).
                # This would not make sense (mean over the pixels?). Instead, we want to
                # log all best videos of all EnvRunners per iteration.
                reduce=None,
                # B/c we do NOT reduce over the video data (mean/min/max), we need to
                # make sure the list of videos in our MetricsLogger does not grow
                # infinitely and gets cleared after each `reduce()` operation, meaning
                # every time, the EnvRunner is asked to send its logged metrics.
                clear_on_reduce=True,
            )
            self.best_episode_and_return = (None, float("-inf"))
        # Worst video.
        if self.worst_episode_and_return[0] is not None:
            metrics_logger.log_value(
                "episode_videos_worst",
                self.worst_episode_and_return[0],
                # Same logging options as above.
                reduce=None,
                clear_on_reduce=True,
            )
            self.worst_episode_and_return = (None, float("inf"))
