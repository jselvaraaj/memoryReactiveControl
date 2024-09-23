from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, DummyVecEnv
import multiprocessing
from featureextractors.gridversefeatureextractor import GridVerseFeatureExtractor
from stable_baselines3 import DQN
import torch

from utils.stubtask import StubTask
from world.worldmaker import WorldMaker
import hydra
from omegaconf import DictConfig
from clearml import Task, Logger
import os
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import logging


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Setup
    torch.set_printoptions(precision=4, sci_mode=False)
    # task = Task.init(project_name='Memory Reactive Control', tags=['DQN Sanity Check'], reuse_last_task_id=False)
    task = StubTask()
    task.connect(cfg)
    torch.manual_seed(cfg.seed)

    # configs
    algorithm_config = cfg.algorithm
    hyperparams_config = cfg.hyperparameters
    environment_config = cfg.environment
    logging_config = cfg.logging

    print("Environment config:", environment_config)
    print("Algorithm config:", algorithm_config)
    print("Hyperparameters:", hyperparams_config)
    print("Logging config:", logging_config)

    # Set up environment
    get_training_env = lambda: TimeLimit(
        WorldMaker.make_env(f'world/{environment_config.gridverse_env}', environment_config),
        max_episode_steps=hyperparams_config.training.max_episode_steps)
    vec_env = make_vec_env(get_training_env, n_envs=1, seed=cfg.seed, vec_env_cls=DummyVecEnv)
    # vec_env = get_training_env()
    vec_env.metadata["render_fps"] = logging_config.testing.video.fps

    train_log_dir = os.path.join('logs', 'train', task.id)
    os.makedirs(os.path.dirname(train_log_dir), exist_ok=True)

    model = DQN('MlpPolicy', vec_env, verbose=2,
                buffer_size=int(algorithm_config.replay_buffer_memory),
                batch_size=int(hyperparams_config.training.batch_size),
                gamma=hyperparams_config.training.discount_factor,
                learning_rate=hyperparams_config.training.learning_rate,
                learning_starts=algorithm_config.training.learning_starts,
                target_update_interval=algorithm_config.training.target_update_interval,
                train_freq=(algorithm_config.training.train_freq, "episode"),
                gradient_steps=algorithm_config.training.gradient_steps,
                exploration_initial_eps=algorithm_config.training.exploration_initial_eps,
                exploration_final_eps=algorithm_config.training.exploration_final_eps,
                seed=cfg.seed,
                tensorboard_log=train_log_dir,
                policy_kwargs={'net_arch': algorithm_config.policy_net.hidden_layers,  # Q network architecture
                               'normalize_images': False,
                               })

    model.learn(total_timesteps=hyperparams_config.training.num_steps,
                log_interval=logging_config.training.log_episode_interval,
                progress_bar=True)

    # upload_video(train_video_folder, 'Training Video')

    print('Training done. Saving model')
    os.makedirs("model_registry", exist_ok=True)
    model_path = os.path.join("model_registry", f"{task.id}_dqn.zip")
    model.save(model_path)
    task.upload_artifact(name='SB3_DQN_Model', artifact_object=model_path)
    vec_env.close()

    print('Testing the model')

    test_log_dir = os.path.join('logs', 'test', f"{task.id}")
    os.makedirs(test_log_dir, exist_ok=True)
    csv_filename = os.path.join(test_log_dir, 'test_results.monitor.csv')
    test_video_folder = os.path.join(test_log_dir, 'videos')

    wrapped_test_env = TimeLimit(WorldMaker.make_env('world/world.yaml', environment_config),
                                 max_episode_steps=algorithm_config.testing.max_episode_steps)
    wrapped_vec_env = make_vec_env(lambda: wrapped_test_env, n_envs=1, seed=cfg.seed, vec_env_cls=DummyVecEnv)

    wrapped_vec_env.metadata["render_fps"] = logging_config.testing.video.fps
    wrapped_vec_env = VecVideoRecorder(wrapped_vec_env, test_video_folder,
                                       record_video_trigger=lambda
                                           x: x % logging_config.testing.video.record_step_interval == 0,
                                       video_length=logging_config.testing.video.length)

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
    task.close()


def upload_video(folder_path: str, name: str):
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):
            file_path = os.path.join(folder_path, filename)
            step = int(filename.split('-')[3])
            Logger.current_logger().report_media('video', name, local_path=file_path, iteration=step)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
