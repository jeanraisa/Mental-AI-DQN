import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import StressReliefEnv

# Create directories for saving models and logs (relative paths)
os.makedirs("models/ppo_models/checkpoints", exist_ok=True)
os.makedirs("models/ppo_models/best", exist_ok=True)
os.makedirs("logs/ppo_logs", exist_ok=True)

# Create environment
env = StressReliefEnv(render_mode=None)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="models/ppo_models/checkpoints/",
    name_prefix="ppo_model"
)

eval_callback = EvalCallback(
    env,
    best_model_save_path="models/ppo_models/best/",
    log_path="logs/ppo_logs/",
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Hyperparameters
hyperparams = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    "verbose": 1
}

# Create and train model
model = PPO(
    "MlpPolicy",
    env,
    tensorboard_log="logs/ppo_logs/tensorboard/",
    **hyperparams
)

model.learn(
    total_timesteps=200000,
    callback=[checkpoint_callback, eval_callback],
    tb_log_name="PPO"
)

# Save final model as ZIP
model.save("models/ppo_models/ppo_stress_relief_final")

# Test the trained model
print("Testing trained PPO model...")
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()
