import multiprocessing
import os

import hydra
from clearml import Logger, Task
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from setup import setup
from utils.stubtask import StubTask
from world.worldmaker import WorldMaker


def setup_testing_env(cfg, environment_config, algorithm_config, logging_config, test_video_folder):
    wrapped_test_env = TimeLimit(
        WorldMaker.make_env(f'./gridverse_conf/{cfg.gridverse_env}', environment_config),
        max_episode_steps=algorithm_config.testing.max_episode_steps)
    wrapped_vec_env = make_vec_env(lambda: wrapped_test_env, n_envs=1, seed=cfg.seed, vec_env_cls=DummyVecEnv)

    wrapped_vec_env.metadata["render_fps"] = logging_config.testing.video.fps
    wrapped_vec_env = VecVideoRecorder(wrapped_vec_env, test_video_folder,
                                       record_video_trigger=lambda
                                           x: x % logging_config.testing.video.record_step_interval == 0,
                                       video_length=logging_config.testing.video.length)
    return wrapped_vec_env


def setup_testing_log(task):
    # test logging setup
    test_log_dir = os.path.join('logs', 'test', f"{task.id}")
    os.makedirs(test_log_dir, exist_ok=True)
    csv_filename = os.path.join(test_log_dir, f'{task.id}_test_results.monitor.csv')
    test_video_folder = os.path.join(test_log_dir, 'videos')

    return test_log_dir, csv_filename, test_video_folder


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    task = Task.init(project_name='Memory Reactive Control', tags=['Testing', cfg.gridverse_env],
                     reuse_last_task_id=False)

    test_model_task_id = "17cbf38ef8a8431e95498961b047ede6"
    test(cfg, test_model_task_id, task)
    task.close()


def test(cfg: DictConfig, test_model_task_id, task=None, ):
    if task is None:
        task = StubTask()

    algorithm_config, hyperparams_config, environment_config, logging_config = setup(task, cfg)

    test_log_dir, csv_log_dir, test_video_folder = setup_testing_log(task)
    wrapped_vec_env = setup_testing_env(cfg, environment_config, algorithm_config, logging_config, test_video_folder)

    print('Loading the model')
    model_name = f"{test_model_task_id}_RecurrentPPO.zip"
    model = RecurrentPPO.load(os.path.join("model_registry", model_name),
                              print_system_info=True)
    task.add_tags([model.__class__.__name__])
    print('Model loaded: ', model_name)

    print('Testing the model')
    mean_reward, std_reward = evaluate_policy(model, wrapped_vec_env,
                                              n_eval_episodes=algorithm_config.testing.n_eval_episodes)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    if not isinstance(task, StubTask):
        upload_video(test_video_folder, 'Test Video')

    print('Testing done')

    # print('Plotting results')
    # results_plotter.plot_results(
    #     [csv_log_dir], num_timesteps=1_0000, x_axis=results_plotter.X_TIMESTEPS, task_name="Gridverse DQN",
    # )
    # plt.show()
    wrapped_vec_env.close()


def upload_video(folder_path: str, name: str):
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):
            file_path = os.path.join(folder_path, filename)
            step = int(filename.split('-')[3])
            Logger.current_logger().report_media('video', name, local_path=file_path, iteration=step)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
