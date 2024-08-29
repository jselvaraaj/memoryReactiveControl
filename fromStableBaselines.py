from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, DummyVecEnv
import multiprocessing
from featureextractors.gridversefeatureextractor import GridVerseFeatureExtractor
from stable_baselines3 import DQN
import torch
from world.worldmaker import WorldMaker
import hydra
from omegaconf import DictConfig
from clearml import Task, Logger
import os
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit


@hydra.main(version_base=None, config_path="conf", config_name="dqntraining")
def main(cfg: DictConfig):
    torch.set_printoptions(precision=4, sci_mode=False)

    task = Task.init(project_name='Memory Reactive Control', tags=['DQN Sanity Check'], reuse_last_task_id=False)
    task.connect(cfg)

    get_training_env = lambda: TimeLimit(WorldMaker.make_env('world/world.yaml'),
                                         max_episode_steps=cfg.training.max_episode_steps)
    vec_env = make_vec_env(get_training_env, n_envs=1, seed=cfg.seed, vec_env_cls=DummyVecEnv)
    # vec_env = get_training_env()

    train_log_dir = os.path.join('logs', 'train', task.id)
    os.makedirs(os.path.dirname(train_log_dir), exist_ok=True)

    train_video_folder = os.path.join(train_log_dir, 'videos')
    vec_env.metadata["render_fps"] = cfg.training.video.fps
    vec_env = VecVideoRecorder(vec_env, train_video_folder,
                               record_video_trigger=lambda x: x % cfg.training.video.record_step_interval == 0,
                               video_length=cfg.training.video.length)

    torch.manual_seed(cfg.seed)
    model = DQN('MultiInputPolicy', vec_env, verbose=2,
                buffer_size=int(cfg.replay_buffer_memory),
                batch_size=int(cfg.training.batch_size),
                gamma=cfg.training.discount_factor,
                learning_rate=cfg.training.learning_rate,
                learning_starts=cfg.training.learning_starts,
                target_update_interval=cfg.training.target_update_interval,
                train_freq=cfg.training.train_freq,
                gradient_steps=cfg.training.gradient_steps,
                exploration_initial_eps=cfg.training.exploration_initial_eps,
                exploration_final_eps=cfg.training.exploration_final_eps,
                seed=cfg.seed,
                tensorboard_log=train_log_dir,
                policy_kwargs={'net_arch': cfg.policy_net.hidden_layers,  # Q network architecture
                               'normalize_images': False,
                               'features_extractor_class': GridVerseFeatureExtractor,
                               'features_extractor_kwargs': {
                                   'grid_embedding_dim': cfg.grid_feature_extraction.embedding_dim,
                                   'cnn_output_dim': cfg.grid_feature_extraction.cnn_output_dim
                               }
                               })

    model.learn(total_timesteps=cfg.training.num_steps, log_interval=cfg.training.log_episode_interval,
                progress_bar=True)

    upload_video(train_video_folder, 'Training Video')

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

    wrapped_test_env = TimeLimit(WorldMaker.make_env('world/world.yaml'),
                                 max_episode_steps=cfg.testing.max_episode_steps)
    wrapped_vec_env = make_vec_env(lambda: wrapped_test_env, n_envs=1, seed=cfg.seed, vec_env_cls=DummyVecEnv)

    wrapped_vec_env.metadata["render_fps"] = cfg.testing.video.fps
    wrapped_vec_env = VecVideoRecorder(wrapped_vec_env, test_video_folder,
                                       record_video_trigger=lambda x: x % cfg.testing.video.record_step_interval == 0,
                                       video_length=cfg.testing.video.length)

    mean_reward, std_reward = evaluate_policy(model, wrapped_vec_env, n_eval_episodes=cfg.testing.n_eval_episodes)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

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
