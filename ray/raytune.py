from ray.rllib.algorithms import PPOConfig


def ppo_with_tune_config(tune_config):
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .framework("torch")
        .learners(num_learners=resources_config.num_learner_workers)
        .resources(
            num_cpus_for_main_process=resources_config.num_cpus_for_main_process,
            num_gpus=resources_config.num_gpus,
        )
        .reporting(log_gradients=True)
        .debugging(
            log_level="DEBUG",
            logger_config={"type": "ray.tune.logger.UnifiedLogger", "logdir": "./logs"},
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                model_config=model_config, catalog_class=GridVersePPOCatalog
            )
        )
        .environment(
            "gridverse",
            env_config={
                "path": f"./gridverse_conf/{env_path}",
                "max_rollout_len": training_config.max_rollout_len,
            },
        )
        .env_runners(
            num_env_runners=env_config.num_env_runners,
            num_envs_per_env_runner=env_config.num_envs_per_env_runner,
            num_cpus_per_env_runner=env_config.num_cpus_per_env_runner,
        )
    )
