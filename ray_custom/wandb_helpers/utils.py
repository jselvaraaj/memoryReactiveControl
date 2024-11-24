import numpy as np
import wandb
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS


def upload_videos_from_result_to_wandb(result):
    video_keys = ['episode_videos_worst', 'episode_videos_best']
    for video_key in video_keys:
        env_runner_result = result[ENV_RUNNER_RESULTS]
        if video_key in env_runner_result:
            videos = env_runner_result[video_key]
            for video in videos:
                frames = np.squeeze(video, axis=0)
                wandb.log({video_key: wandb.Video(frames, fps=4)})
