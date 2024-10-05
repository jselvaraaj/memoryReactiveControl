import multiprocessing
import os

import hydra
import torch
from clearml import Logger, Task
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from featureextractors.gridversefeatureextractor import GridVerseFeatureExtractor
from utils.stubtask import StubTask
from world.worldmaker import WorldMaker


def setup_training_env(cfg, environment_config, hyperparams_config, logging_config):
    # Set up environment
    def get_training_env():
        gridverse_env = WorldMaker.make_env(f'./gridverse_conf/{environment_config.gridverse_env}', environment_config)
        return TimeLimit(gridverse_env, max_episode_steps=hyperparams_config.training.max_episode_steps)

    vec_env = make_vec_env(get_training_env, n_envs=environment_config.number_of_envs_to_run_parallelly, seed=cfg.seed,
                           vec_env_cls=DummyVecEnv)
    # vec_env = get_training_env()
    vec_env.metadata["render_fps"] = logging_config.testing.video.fps

    return vec_env


def setup_training_logging(task):
    # train logging setup
    train_log_dir = os.path.join('logs', 'train', task.id)
    os.makedirs(os.path.dirname(train_log_dir), exist_ok=True)

    # test logging setup
    test_log_dir = os.path.join('logs', 'test', f"{task.id}")
    os.makedirs(test_log_dir, exist_ok=True)
    csv_filename = os.path.join(test_log_dir, f'{task.id}_test_results.monitor.csv')
    test_video_folder = os.path.join(test_log_dir, 'videos')

    return train_log_dir, test_log_dir, csv_filename, test_video_folder


def setup_testing_env(cfg, environment_config, algorithm_config, logging_config, test_video_folder):
    wrapped_test_env = TimeLimit(
        WorldMaker.make_env(f'./gridverse_conf/{environment_config.gridverse_env}', environment_config),
        max_episode_steps=algorithm_config.testing.max_episode_steps)
    wrapped_vec_env = make_vec_env(lambda: wrapped_test_env, n_envs=1, seed=cfg.seed, vec_env_cls=DummyVecEnv)

    wrapped_vec_env.metadata["render_fps"] = logging_config.testing.video.fps
    wrapped_vec_env = VecVideoRecorder(wrapped_vec_env, test_video_folder,
                                       record_video_trigger=lambda
                                           x: x % logging_config.testing.video.record_step_interval == 0,
                                       video_length=logging_config.testing.video.length)
    return wrapped_vec_env


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Setup
    torch.set_printoptions(precision=4, sci_mode=False)
    task = Task.init(project_name='Memory Reactive Control', tags=['RecurrentPPO Sanity Check'],
                     reuse_last_task_id=False)
    # task = StubTask()
    task.connect(cfg)
    torch.manual_seed(cfg.seed)

    # configs
    algorithm_config = cfg.algorithm
    hyperparams_config = cfg.hyperparameters
    environment_config = cfg.environment
    logging_config = cfg.logging

    print("Environment config:", environment_config, "\n")
    print("Algorithm config:", algorithm_config, "\n")
    print("Hyperparameters:", hyperparams_config, "\n")
    print("Logging config:", logging_config, "\n")

    vec_env = setup_training_env(cfg, environment_config, hyperparams_config, logging_config)
    train_log_dir, test_log_dir, csv_filename, test_video_folder = setup_training_logging(task)

    model = RecurrentPPO('MultiInputLstmPolicy', vec_env, verbose=2, seed=cfg.seed,
                         learning_rate=hyperparams_config.training.learning_rate,
                         n_steps=hyperparams_config.training.num_env_steps_for_each_gradient_update,
                         batch_size=int(hyperparams_config.training.batch_size),
                         n_epochs=hyperparams_config.training.num_epochs_optimizing_surrogate_loss,
                         gamma=hyperparams_config.training.discount_factor,
                         gae_lambda=hyperparams_config.training.gae_lambda,
                         clip_range=hyperparams_config.training.clip_range,
                         clip_range_vf=hyperparams_config.training.clip_range_vf,
                         normalize_advantage=hyperparams_config.training.normalize_advantage,
                         ent_coef=hyperparams_config.training.ent_coef,
                         vf_coef=hyperparams_config.training.vf_coef,
                         max_grad_norm=hyperparams_config.training.max_grad_norm,
                         target_kl=hyperparams_config.training.target_kl,
                         stats_window_size=logging_config.training.stats_episode_window_size,
                         tensorboard_log=train_log_dir,
                         use_sde=algorithm_config.use_sde,
                         policy_kwargs={
                             'activation_fn': torch.nn.ReLU if algorithm_config.activation_fn == 'ReLU' else torch.nn.Tanh,
                             'use_expln': algorithm_config.use_expln,

                             # Feature extractors
                             'share_features_extractor': algorithm_config.feature_extractor.share_feature_extractor,
                             'features_extractor_class': GridVerseFeatureExtractor,
                             'features_extractor_kwargs': {
                                 'config': algorithm_config.feature_extractor,
                             },
                             'normalize_images': False,

                             # LSTM
                             'enable_critic_lstm': algorithm_config.lstm.enable_critic_lstm,
                             'shared_lstm': algorithm_config.lstm.shared_lstm,
                             'lstm_hidden_size': algorithm_config.lstm.lstm_hidden_size,
                             'n_lstm_layers': algorithm_config.lstm.n_lstm_layers,

                             # MLP extractors
                             'net_arch': {
                                 'pi': algorithm_config.mlp.policy_net.lstm_output_to_latent_features,
                                 'vf': algorithm_config.mlp.value_net.lstm_output_to_latent_features
                             },
                         }
                         )

    model.learn(total_timesteps=hyperparams_config.training.total_num_steps,
                log_interval=logging_config.training.log_episode_interval,
                progress_bar=True)

    # model = RecurrentPPO.load(os.path.join("model_registry", "48b251396ac1470887068cfe21bec887_RecurrentPPO.zip"),
    #                           print_system_info=True)

    # upload_video(train_video_folder, 'Training Video')

    print('Training done. Saving model')
    os.makedirs("model_registry", exist_ok=True)
    model_path = os.path.join("model_registry", f"{task.id}_{model.__class__.__name__}.zip")
    model.save(model_path)
    task.upload_artifact(name=f'SB3_{model.__class__.__name__}_Model', artifact_object=model_path)
    vec_env.close()

    print('Testing the model')
    wrapped_vec_env = setup_testing_env(cfg, environment_config, algorithm_config, logging_config, test_video_folder)
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
