"""
ClearML Remote Training Script for OT2 RL
This script runs grid search training on the school's ClearML server.

Usage:
    python train_clearml.py --member member2 --max_configs 20
"""

import argparse
import sys
import os
from pathlib import Path

# ============================================================================
# ClearML INITIALIZATION - MUST BE AT THE TOP
# ============================================================================
from clearml import Task

# Parse arguments first (before ClearML takes over)
parser = argparse.ArgumentParser(description='Run RL training on ClearML server')
parser.add_argument('--member', type=str, default='member2',
                    choices=['member1', 'member2', 'member3', 'member4'],
                    help='Which member config to run (member1=Ana, member2=Floris, member3=Raya, member4=Stijn)')
parser.add_argument('--max_configs', type=int, default=20,
                    help='Maximum number of configurations to test')
parser.add_argument('--total_timesteps', type=int, default=2_000_000,
                    help='Total timesteps per training run')
parser.add_argument('--project_name', type=str, default='OT2-RL',
                    help='ClearML project name')
args = parser.parse_args()

# Get member name for task naming
MEMBER_NAMES = {
    'member1': 'Ana',
    'member2': 'Floris',
    'member3': 'Raya',
    'member4': 'Stijn'
}
member_name = MEMBER_NAMES[args.member]

# Add package requirements BEFORE Task.init()
Task.add_requirements("numpy", "1.24.3")
Task.add_requirements("pandas", "2.0.3")
Task.add_requirements("wandb")
Task.add_requirements("stable-baselines3")
Task.add_requirements("gymnasium")
Task.add_requirements("pybullet")

# Initialize ClearML task
task = Task.init(
    project_name=f'{args.project_name}/{member_name}',
    task_name=f'{member_name}_GridSearch'
)

# Set the base docker image (provided by the school)
task.set_base_docker('deanis/2023y2b-rl:latest')

# Execute remotely on the default queue
task.execute_remotely(queue_name="default")

# ============================================================================
# PATH SETUP - All files are in the same directory for this public repo
# ============================================================================
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))
os.chdir(script_dir)
print(f"[INFO] Working directory: {os.getcwd()}")

# ============================================================================
# W&B API KEY - Set your API key here
# Get your key from: https://wandb.ai/authorize
# ============================================================================
os.environ['WANDB_API_KEY'] = '7b7a17f3abfaefa9ee08ea72387f76ee86e79318'

# ============================================================================
# IMPORTS
# ============================================================================
import itertools
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback,
    StopTrainingOnRewardThreshold
)
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

from sim_class import Simulation
from ot2_gym_wrapper import OT2GymEnv

# ============================================================================
# CONFIGURATION (same as notebook)
# ============================================================================
HYPERPARAMETER_RANGES = {
    "algorithm": ["PPO", "SAC", "TD3"],
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
    "action_scale": [0.01, 0.05, 0.1, 0.2],
    "max_steps": [300, 500, 1000],
    "distance_weight": [1.0, 5.0, 10.0],
    "progress_weight": [1.0, 5.0, 10.0, 20.0],
    "success_bonus": [50.0, 100.0, 200.0],
    "time_penalty": [0.0, 0.01, 0.05],
    "ppo_n_steps": [512, 1024, 2048, 4096],
    "ppo_batch_size": [32, 64, 128, 256],
    "ppo_n_epochs": [5, 10, 20],
    "ppo_gamma": [0.95, 0.99, 0.995],
    "ppo_gae_lambda": [0.9, 0.95, 0.98],
    "ppo_clip_range": [0.1, 0.2, 0.3],
    "ppo_ent_coef": [0.0, 0.001, 0.01],
    "buffer_size": [100_000, 500_000, 1_000_000],
    "sac_batch_size": [64, 128, 256],
    "sac_gamma": [0.95, 0.99, 0.995],
    "sac_tau": [0.001, 0.005, 0.01],
    "sac_ent_coef": ["auto", 0.1, 0.2],
}

GROUP_ASSIGNMENTS = {
    "member1": {
        "name": "Ana",
        "focus": "Learning Rate & Action Scale",
        "hyperparameters": ["algorithm", "learning_rate", "action_scale"],
        "fixed": {"max_steps": 500, "ppo_n_steps": 2048, "ppo_batch_size": 64}
    },
    "member2": {
        "name": "Floris",
        "focus": "PPO Architecture",
        "hyperparameters": ["ppo_n_steps", "ppo_batch_size", "ppo_n_epochs", "ppo_gamma"],
        "fixed": {"algorithm": "PPO", "learning_rate": 3e-4, "action_scale": 0.1, "max_steps": 500}
    },
    "member3": {
        "name": "Raya",
        "focus": "Reward Function",
        "hyperparameters": ["distance_weight", "progress_weight", "success_bonus", "time_penalty"],
        "fixed": {"algorithm": "PPO", "learning_rate": 3e-4, "action_scale": 0.1, "max_steps": 500}
    },
    "member4": {
        "name": "Stijn",
        "focus": "Environment & Control Mode",
        "hyperparameters": ["max_steps", "action_scale"],
        "fixed": {"algorithm": "SAC", "learning_rate": 3e-4, "buffer_size": 1_000_000}
    },
}

def get_base_config():
    return {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "total_timesteps": args.total_timesteps,
        "max_steps": 500,
        "action_scale": 0.1,
        "success_threshold": 0.001,
        "distance_weight": 10.0,
        "progress_weight": 50.0,
        "success_bonus": 100.0,
        "time_penalty": 0.01,
        "ppo_n_steps": 2048,
        "ppo_batch_size": 64,
        "ppo_n_epochs": 10,
        "ppo_gamma": 0.99,
        "ppo_gae_lambda": 0.95,
        "ppo_clip_range": 0.2,
        "ppo_ent_coef": 0.0,
        "ppo_vf_coef": 0.5,
        "ppo_max_grad_norm": 0.5,
        "buffer_size": 1_000_000,
        "learning_starts": 100,
        "sac_batch_size": 256,
        "sac_gamma": 0.99,
        "sac_tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "sac_ent_coef": "auto",
        "td3_policy_delay": 2,
        "use_early_stopping": True,
        "reward_threshold": 90.0,
        "eval_freq": 10_000,
        "n_eval_episodes": 10,
        "checkpoint_freq": 50_000,
        "model_save_freq": 10_000,
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def generate_grid_search_configs(member_id: str, max_configs: int = 20):
    assignment = GROUP_ASSIGNMENTS[member_id]
    hp_to_search = assignment["hyperparameters"]
    ranges = {hp: HYPERPARAMETER_RANGES[hp] for hp in hp_to_search}

    keys = list(ranges.keys())
    values = list(ranges.values())
    combinations = list(itertools.product(*values))

    if len(combinations) > max_configs:
        print(f"Total combinations: {len(combinations)}, sampling {max_configs}")
        indices = np.random.choice(len(combinations), max_configs, replace=False)
        combinations = [combinations[i] for i in indices]

    configs = []
    for combo in combinations:
        config = assignment["fixed"].copy()
        for key, value in zip(keys, combo):
            config[key] = value
        configs.append(config)

    return configs

def merge_configs(base_config: dict, overrides: dict) -> dict:
    config = base_config.copy()
    config.update(overrides)
    return config

def make_env(render_mode=None, **kwargs):
    env = OT2GymEnv(render_mode=render_mode, **kwargs)
    env = Monitor(env)
    return env

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_rl_agent(config: dict, run_name: str = None):
    run = wandb.init(
        project="ot2-rl-controller",
        entity="upu-one",
        name=run_name,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/{run.id}_{timestamp}"
    log_dir = f"logs/{run.id}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFORM] run ID: {run.id}")
    for key, value in config.items():
        print(f"[INFORM] {key}: {value}")

    print("\n[TRYING] creating environments")
    train_env = make_env(
        render_mode=None,
        max_steps=config["max_steps"],
        action_scale=config["action_scale"],
        success_threshold=config["success_threshold"]
    )

    eval_env = make_env(
        render_mode=None,
        max_steps=config["max_steps"],
        action_scale=config["action_scale"],
        success_threshold=config["success_threshold"]
    )
    print("[RESULT] created environments")

    algorithm = config["algorithm"]
    print(f"[TRYING] initializing {algorithm} agent")

    if algorithm == "PPO":
        model = PPO(
            policy=config["policy"],
            env=train_env,
            learning_rate=config["learning_rate"],
            n_steps=config["ppo_n_steps"],
            batch_size=config["ppo_batch_size"],
            n_epochs=config["ppo_n_epochs"],
            gamma=config["ppo_gamma"],
            gae_lambda=config["ppo_gae_lambda"],
            clip_range=config["ppo_clip_range"],
            ent_coef=config["ppo_ent_coef"],
            vf_coef=config["ppo_vf_coef"],
            max_grad_norm=config["ppo_max_grad_norm"],
            tensorboard_log=log_dir,
            verbose=1
        )
    elif algorithm == "SAC":
        model = SAC(
            policy=config["policy"],
            env=train_env,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            learning_starts=config["learning_starts"],
            batch_size=config["sac_batch_size"],
            tau=config["sac_tau"],
            gamma=config["sac_gamma"],
            train_freq=config["train_freq"],
            gradient_steps=config["gradient_steps"],
            ent_coef=config["sac_ent_coef"],
            tensorboard_log=log_dir,
            verbose=1
        )
    elif algorithm == "TD3":
        model = TD3(
            policy=config["policy"],
            env=train_env,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            learning_starts=config["learning_starts"],
            batch_size=config["sac_batch_size"],
            tau=config["sac_tau"],
            gamma=config["sac_gamma"],
            train_freq=config["train_freq"],
            gradient_steps=config["gradient_steps"],
            policy_delay=config["td3_policy_delay"],
            tensorboard_log=log_dir,
            verbose=1
        )
    else:
        raise ValueError(f"[ERRORS] unknown algorithm: {algorithm}")

    print(f"[RESULT] initialized {algorithm} agent")

    callbacks = []

    wandb_callback = WandbCallback(
        model_save_freq=config["model_save_freq"],
        model_save_path=save_dir,
        verbose=2
    )
    callbacks.append(wandb_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="ot2_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)

    if config["use_early_stopping"]:
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=config["reward_threshold"],
            verbose=1
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=callback_on_best,
            eval_freq=config["eval_freq"],
            n_eval_episodes=config["n_eval_episodes"],
            best_model_save_path=os.path.join(save_dir, "best_model"),
            log_path=log_dir,
            deterministic=True,
            render=False,
            verbose=1,
        )
        callbacks.append(eval_callback)

    callback = CallbackList(callbacks)

    print("[TRYING] starting training")
    print(f"[INFORM] total timesteps: {config['total_timesteps']:,}")

    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("[ERRORS] training interrupted by user")

    final_model_path = os.path.join(save_dir, "final_model")
    model.save(final_model_path)
    print(f"[RESULT] final model saved to: {final_model_path}")

    # Upload to ClearML
    task.upload_artifact("final_model", artifact_object=f"{final_model_path}.zip")

    print("\n[TRYING] quick evaluation")
    n_eval_episodes = 20
    episode_rewards = []
    success_count = 0

    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)
        if info.get('is_success', False):
            success_count += 1

    final_metrics = {
        "final/mean_reward": np.mean(episode_rewards),
        "final/success_rate": success_count / n_eval_episodes,
        "final/n_episodes": n_eval_episodes,
    }
    wandb.log(final_metrics)

    print(f"[RESULT] completed evaluation ({n_eval_episodes} episodes)")
    print(f"[INFORM] mean reward: {final_metrics['final/mean_reward']:.2f}")
    print(f"[INFORM] success rate: {final_metrics['final/success_rate']*100:.1f}%")

    wandb.save(f"{final_model_path}.zip")

    train_env.close()
    eval_env.close()
    wandb.finish()

    print("\n[RESULT] finished training")
    return final_metrics, save_dir

# ============================================================================
# GRID SEARCH
# ============================================================================
def run_grid_search(member_id: str, max_configs: int = 20):
    assignment = GROUP_ASSIGNMENTS[member_id]

    print(f"[INFORM] running grid search for {assignment['name']}")
    print(f"[INFORM] focus: {assignment['focus']}")
    print(f"[INFORM] hyperparameters: {assignment['hyperparameters']}")
    print(f"[INFORM] fixed parameters: {assignment['fixed']}\n")

    print("[TRYING] generating configurations...")
    configs = generate_grid_search_configs(member_id, max_configs)
    base_config = get_base_config()
    print(f"[RESULT] generated {len(configs)} configurations to test")

    results = []

    for i, override_config in enumerate(configs):
        print(f"\n[TRYING] starting configuration {i+1}/{len(configs)}...")

        config = merge_configs(base_config, override_config)
        run_name = f"{assignment['name']}_config{i+1}"

        metrics, save_dir = train_rl_agent(config, run_name)

        result = {
            "config_num": i + 1,
            "config": override_config,
            "metrics": metrics,
            "save_dir": save_dir,
        }
        results.append(result)

        print(f"[RESULT] completed configuration {i+1}/{len(configs)}")

    print("[RESULT] grid search complete")

    results.sort(key=lambda x: x['metrics']['final/success_rate'], reverse=True)

    print("\n[TRYING] summarizing top 5 configurations:")
    for i, result in enumerate(results[:5]):
        print(f"\n[RESULT] configuration {i+1}: {result['config_num']}")
        print(f"[INFORM] success rate: {result['metrics']['final/success_rate']*100:.1f}%")
        print(f"[INFORM] mean reward: {result['metrics']['final/mean_reward']:.2f}")
        print(f"[INFORM] hyperparameters: {result['config']}")

    return results

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Login to W&B
    wandb.login()

    print(f"\n{'='*60}")
    print(f"Running grid search for: {member_name} ({args.member})")
    print(f"Max configs: {args.max_configs}")
    print(f"Timesteps per run: {args.total_timesteps:,}")
    print(f"{'='*60}\n")

    # Run grid search
    results = run_grid_search(args.member, args.max_configs)

    print("\n[DONE] All training complete!")
