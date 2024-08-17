import torch
from world.worldmaker import WorldMaker
from networks.dqn import DQN
from training.dqntrainer import DQNTrainer

if __name__ == '__main__':
    world = WorldMaker.make_env('world/world.yaml')
    world.reset()
    state_shape = torch.cat([torch.flatten(torch.tensor(s)) for s in world.state.values()]).shape[0]
    model = DQN(state_shape, world.action_space.n)

    trainer = DQNTrainer(world, model)
    trainer.train()
