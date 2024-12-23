# Partial Observable Grid World Experiments

This repository contains experiments for partially observable grid world environments in [**GridVerse**](https://github.com/abaisero/gym-gridverse) with **Stable Baselines**, and **RLlib**.

![Key door POMDP](https://raw.githubusercontent.com/jselvaraaj/memoryReactiveControl/303b6020e47546e92fc2cc3b50eb58b8bfd6690a/key_door.gif)

## Key Feature

Custom networks for gridverse dict observations. This network at a high level is a CNN followed by an optional RNN. To make this work, custom encoder where implmeneted in stabelines and custom catlog, connectors and encoders in rllib. Hydra is used for config management.

## Directory structure
- config (top level) is for configuring gridverse environment (rewards, states, observations and such) and for configuring the algorithm and hyperparameters.
- ray_custom/experiment_manager.py is the driver code for running PPO( easily configurable to other algorithms from rllib) in gridverse with rllib
- stablebaselines_exp/stablebaselines_exp/stablebaselinestrain.py is the driver code for training PPO( easily configurable to other algorithms from stablebaseline) in grifverse with stablebaseline
- gridverse_torch_featureextractors are pytorch networks that are used as building pieces in both rllib and stablebaselines.
- gridverse_utils contains some utility functions for creating the gridverse environment as a gym environment.
