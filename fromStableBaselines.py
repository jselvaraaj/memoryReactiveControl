import multiprocessing

import hydra
from clearml import Task
from omegaconf import DictConfig

from test import test
from train import train


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    task = Task.init(project_name='Memory Reactive Control',
                     tags=['Training', 'Testing', cfg.gridverse_env],
                     reuse_last_task_id=False)
    train(cfg, task)
    test(cfg, task.id, task)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
