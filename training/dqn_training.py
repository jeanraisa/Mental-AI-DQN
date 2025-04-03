import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from environment.custom_env import StressReliefEnv

# Create directories
os.makedirs("models/dqn_models/checkpoints", exist_ok=True)
os.makedirs("models/dqn_models/", exist_ok=True)
os.makedirs("logs/dqn_logs", exist_ok=True)

# Create environment with easier settings for initial training
env = StressReliefEnv(
    render_mode=None,
    zone_move_interval=100,  # Reduced zone movement frequency
    obstacle_penalty=-5,     # Reduced penalty
    static_obstacles=[       # Simplified obstacle layout
        [300, 200],
        [500, 400]
    ]
)
env = Monitor(env)
env = DummyVecEnv([lambda: env])
# Optional: Stack frames for temporal awareness
# env = VecFrameStack(env, n_stack=3)

# Enhanced callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=25000,         # More frequent checkpoints
    save_path="models/dqn_models/checkpoints/",
    name_prefix="dqn_model",
    verbose=1
)

eval_callback = EvalCallback(
    env,
    best_model_save_path="models/dqn_models/",
    log_path="logs/dqn_logs/",
    eval_freq=10000,         # More evaluation points
    n_eval_episodes=5,      # More reliable evaluation
    deterministic=True,
    render=False,
    verbose=1
)

# Optimized hyperparameters
hyperparams = {
    "learning_rate": 3e-4,   # Increased learning rate
    "buffer_size": 50000,   # Larger replay buffer
    "learning_starts": 5000, # Earlier learning start
    "batch_size": 64,       # Larger batch size
    "gamma": 0.95,          # Slightly lower discount
    "target_update_interval": 500, # More frequent target updates
    "train_freq": 4,
    "gradient_steps": 1,
    "exploration_fraction": 0.3, # Longer exploration
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.01, # Lower final exploration
    "policy_kwargs": {
        "net_arch": [128, 64] # Deeper network
    },
    "verbose": 1
}

# Create model
model = DQN(
    "MlpPolicy",
    env,
    tensorboard_log="logs/dqn_logs/tensorboard/",
    **hyperparams
)

# Extended training
model.learn(
    total_timesteps=200000,  # Longer training
    callback=[checkpoint_callback, eval_callback],
    tb_log_name="DQN",
    progress_bar=True
)

# Save final model
model.save("models/dqn_models/dqn_stress_relief_final")

# Test with rendering
print("Testing trained model with visualization...")
test_env = StressReliefEnv(render_mode="human")
obs = test_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = test_env.step(action)
    if done:
        obs = test_env.reset()
test_env.close()
