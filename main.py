from world.worldmaker import WorldMaker
from networks.dqn import DQN
from training.dqntrainer import DQNTrainer

if __name__ == '__main__':
    world = WorldMaker.make_env('world/world.yaml')
    world.reset()
    print(world.observation_space.shape, world.state_space.shape, world.action_space.n)
    # model = DQN(world.observation_space.shape[0], world.action_space.n)
    #
    # trainer = DQNTrainer(world, model)
    # trainer.train()
