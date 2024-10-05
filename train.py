import multiprocessing
import os

import hydra
import torch
from clearml import Task
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from featureextractors.gridversefeatureextractor import GridVerseFeatureExtractor
from setup import setup
from utils.stubtask import StubTask
from world.worldmaker import WorldMaker


def setup_training_env(cfg, environment_config, hyperparams_config, logging_config):
    # Set up environment
    def get_training_env():
        gridverse_env = WorldMaker.make_env(f'./gridverse_conf/{cfg.gridverse_env}', environment_config)
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

    return train_log_dir


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    task = Task.init(project_name='Memory Reactive Control', tags=['Training', cfg.gridverse_env],
                     reuse_last_task_id=False)
    # task = StubTask()
    train(cfg, task)
    task.close()


def train(cfg: DictConfig, task=None):
    if task is None:
        task = StubTask()

    algorithm_config, hyperparams_config, environment_config, logging_config = setup(task, cfg)

    vec_env = setup_training_env(cfg, environment_config, hyperparams_config, logging_config)
    train_log_dir = setup_training_logging(task)

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
    task.add_tags([model.__class__.__name__])

    model.learn(total_timesteps=hyperparams_config.training.total_num_steps,
                log_interval=logging_config.training.log_episode_interval,
                progress_bar=True)

    print('Training done. Saving model')
    os.makedirs("model_registry", exist_ok=True)
    model_path = os.path.join("model_registry", f"{task.id}_{model.__class__.__name__}.zip")
    model.save(model_path)
    task.upload_artifact(name=f'SB3_{model.__class__.__name__}_Model', artifact_object=model_path)
    vec_env.close()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
