from common.env_util import make_vec_env
from common.evaluation import evaluate_policy
from common.vec_env import SubprocVecEnv
from matplotlib import pyplot as plt
from utils.featureextractor import GridVerseFeatureExtractor
from stable_baselines3 import DQN
import torch
from world.worldmaker import WorldMaker
import hydra
from omegaconf import DictConfig
from clearml import Task
import os
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit


@hydra.main(version_base=None, config_path="conf", config_name="dqntraining")
def main(cfg: DictConfig):
    torch.set_printoptions(precision=4, sci_mode=False)

    task = Task.init(project_name='Memory Reactive Control', tags=['DQN Sanity Check'])
    task.connect(cfg)

    get_training_env = lambda: TimeLimit(WorldMaker.make_env('world/world.yaml'),
                                         max_episode_steps=cfg.training.max_episode_steps)
    vec_env = make_vec_env(get_training_env, n_envs=4, seed=cfg.seed, vec_env_cls=SubprocVecEnv)
    # vec_env = get_training_env()
    train_log_dir = os.path.join('logs', 'train', task.id)
    os.makedirs(os.path.dirname(train_log_dir), exist_ok=True)

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

    print('Training done. Saving model')
    os.makedirs("model_registry", exist_ok=True)
    model.save(os.path.join("model_registry", f"{task.id}_dqn"))
    vec_env.close()

    print('Testing the model')

    csv_log_dir = os.path.join('logs', 'test', f"{task.id}")
    os.makedirs(csv_log_dir, exist_ok=True)
    csv_filename = os.path.join(csv_log_dir, 'test_results.monitor.csv')
    wrapped_env = Monitor(TimeLimit(TimeLimit(WorldMaker.make_env('world/world.yaml'),
                                              max_episode_steps=cfg.testing.max_episode_steps),
                                    max_episode_steps=cfg.testing.max_episode_steps),
                          filename=f"{csv_filename}")

    mean_reward, std_reward = evaluate_policy(model, wrapped_env, n_eval_episodes=cfg.testing.n_eval_episodes)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    print('Testing done')

    print('Plotting results')
    results_plotter.plot_results(
        [csv_log_dir], num_timesteps=1_0000, x_axis=results_plotter.X_TIMESTEPS, task_name="Gridverse DQN",
    )
    plt.show()

    task.close()


if __name__ == '__main__':
    main()
