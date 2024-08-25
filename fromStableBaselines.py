from common.evaluation import evaluate_policy
from utils.featureextractor import GridVerseFeatureExtractor
from stable_baselines3 import DQN
import torch
from world.worldmaker import WorldMaker
import hydra
from omegaconf import DictConfig
from clearml import Task
import os


@hydra.main(version_base=None, config_path="conf", config_name="dqntraining")
def main(cfg: DictConfig):
    torch.set_printoptions(precision=4, sci_mode=False)

    task = Task.init(project_name='Memory Reactive Control', tags=['DQN Sanity Check'])
    task.connect(cfg)

    world = WorldMaker.make_env('world/world.yaml')
    obs, _ = world.reset()

    torch.manual_seed(cfg.seed)
    model = DQN('MultiInputPolicy', world, verbose=2,
                buffer_size=cfg.replay_buffer_memory,
                batch_size=cfg.training.batch_size,
                gamma=cfg.training.discount_factor,
                learning_rate=cfg.training.learning_rate,
                learning_starts=cfg.training.learning_starts,
                target_update_interval=cfg.training.target_update_interval,
                exploration_initial_eps=cfg.training.exploration_initial_eps,
                exploration_final_eps=cfg.training.exploration_final_eps,
                seed=cfg.seed,
                tensorboard_log='logs/',
                policy_kwargs={'net_arch': cfg.policy_net.hidden_layers,  # Q network architecture
                               'normalize_images': False,
                               'features_extractor_class': GridVerseFeatureExtractor,
                               'features_extractor_kwargs': {
                                   'grid_embedding_dim': cfg.grid_feature_extraction.embedding_dim,
                                   'cnn_output_dim': cfg.grid_feature_extraction.cnn_output_dim
                               }
                               })

    model.learn(total_timesteps=cfg.training.num_steps, log_interval=cfg.training.log_interval, progress_bar=True)

    print('Training done. Saving model')
    os.makedirs("model_registry", exist_ok=True)
    model.save(os.path.join("model_registry", f"{task.id}_dqn"))

    print('Testing the model')
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print('Testing done')
    world.close()
    task.close()


if __name__ == '__main__':
    main()
