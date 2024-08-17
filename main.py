import os
from tempfile import gettempdir

import torch
from dqn.dqntester import DQNTester
from world.worldmaker import WorldMaker
from dqn.networks.dqn import DQN
from dqn.dqntrainer import DQNTrainer
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="dqntraining")
def main(cfg: DictConfig):
    world = WorldMaker.make_env('world/world.yaml')
    world.reset()
    state_shape = torch.cat([torch.flatten(torch.tensor(s)) for s in world.state.values()]).shape[0]

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if cfg.use_cuda and torch.cuda.is_available() else "cpu")

    model = DQN(state_shape, world.action_space.n, cfg.policy_net.hidden_layers, device)

    trainer = DQNTrainer(world, model, cfg)
    trainer.train()

    print('Training done. Saving model')
    torch.save(model.state_dict(), os.path.join(gettempdir(), "dqn.pt"))

    print('Testing the model')
    tester = DQNTester(world, model, cfg)
    tester.test()
    print('Testing done')


if __name__ == '__main__':
    main()
