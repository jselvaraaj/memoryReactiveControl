from world.worldmaker import WorldMaker
from networks.dqn import DQN
from training.dqntrainer import DQNTrainer

if __name__ == '__main__':
    world = WorldMaker.make_env('world/world.yaml')
    model = DQN(world.observation_space.shape[0], world.action_space.n)

    trainer = DQNTrainer(world, model)
    trainer.train()
