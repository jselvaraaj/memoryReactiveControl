import sys
import torch
from dqn.dqntester import DQNTester
from world.worldmaker import WorldMaker
from dqn.networks.dqn import DQN
from dqn.dqntrainer import DQNTrainer
import hydra
from omegaconf import DictConfig
from clearml import Task
import os


@hydra.main(version_base=None, config_path="conf", config_name="dqntraining")
def main(cfg: DictConfig):
    torch.set_printoptions(precision=4, sci_mode=False)

    Task.force_requirements_env_freeze(
        requirements_file=os.path.join(hydra.utils.get_original_cwd(), 'requirements.txt'))
    task = Task.init(project_name='Memory Reactive Control', tags=['DQN Sanity Check'])
    task.connect(cfg)

    world = WorldMaker.make_env('world/world.yaml')
    world.reset()
    state_shape = torch.cat([torch.flatten(torch.tensor(s)) for s in world.state.values()]).shape[0]

    torch.manual_seed(cfg.seed)
    device = None
    if cfg.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif cfg.use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = DQN(state_shape, world.action_space.n, cfg, device)

    trainer = DQNTrainer(world, model, cfg, device)
    trainer.train()

    print('Training done. Saving model')
    os.makedirs("model_registry", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("model_registry", f"{task.id}_dqn.pt"))

    print('Testing the model')
    tester = DQNTester(world, model, cfg, device)
    tester.test()
    print('Testing done')
    world.close()
    task.close()


if __name__ == '__main__':
    main()
    sys.exit(0)
