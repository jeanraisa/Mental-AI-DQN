import os
import time
import random
import numpy as np
import pygame
from pygame.locals import *
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env, spaces

# Configuration
MODEL_TYPE = "dqn" #Select "dqn" or "ppo"
USE_MODEL = False  # Start with random actions for testing
NUM_EPISODES = 10
GRID_SIZE = 5
CELL_SIZE = 100
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
ZONE_MOVE_INTERVAL = 2  # seconds
REST_INTERVAL_EPISODES = 2  # Rest every 2 episodes
REST_DURATION = 4  # seconds

class GridEnvironment(Env):
    def __init__(self):
        super(GridEnvironment, self).__init__()
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=GRID_SIZE-1, shape=(2,), dtype=np.int32)
        
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.agent_pos = [0, 0]
        self.obstacles = []
        self.safe_zones = []
        self.meditation_zone = []
        self.last_zone_move_time = time.time()
        
        # Visualization
        self.render_mode = "human"
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
            pygame.display.set_caption("Mental AI - Grid Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 16)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.agent_pos = [random.randint(0, GRID_SIZE-1), 
                         random.randint(0, GRID_SIZE-1)]
        
        self.obstacles = []
        num_obstacles = random.randint(3, 5)
        while len(self.obstacles) < num_obstacles:
            pos = [random.randint(0, GRID_SIZE-1), 
                  random.randint(0, GRID_SIZE-1)]
            if pos != self.agent_pos and pos not in self.obstacles:
                self.obstacles.append(pos)
        
        self._generate_zones()
        self.last_zone_move_time = time.time()
        
        return np.array(self.agent_pos, dtype=np.int32), {}
    
    def _generate_zones(self):
        self.safe_zones = []
        self.meditation_zone = []
        
        num_safe = random.randint(2, 3)
        while len(self.safe_zones) < num_safe:
            pos = [random.randint(0, GRID_SIZE-1), 
                  random.randint(0, GRID_SIZE-1)]
            if (pos != self.agent_pos and 
                pos not in self.obstacles and 
                pos not in self.safe_zones):
                self.safe_zones.append(pos)
        
        while not self.meditation_zone:
            pos = [random.randint(0, GRID_SIZE-1), 
                  random.randint(0, GRID_SIZE-1)]
            if (pos != self.agent_pos and 
                pos not in self.obstacles and 
                pos not in self.safe_zones):
                self.meditation_zone = pos
    
    def _check_zone_movement(self):
        current_time = time.time()
        if current_time - self.last_zone_move_time >= ZONE_MOVE_INTERVAL:
            self.last_zone_move_time = current_time
            return True
        return False
    
    def step(self, action):
        if self._check_zone_movement():
            self._generate_zones()
        
        prev_pos = self.agent_pos.copy()
        
        if action == 0:   # Up
            self.agent_pos[1] = max(0, self.agent_pos[1]-1)
        elif action == 1: # Down
            self.agent_pos[1] = min(GRID_SIZE-1, self.agent_pos[1]+1)
        elif action == 2: # Left
            self.agent_pos[0] = max(0, self.agent_pos[0]-1)
        elif action == 3: # Right
            self.agent_pos[0] = min(GRID_SIZE-1, self.agent_pos[0]+1)
        
        reward = -0.05
        done = False
        truncated = False
        info = {'hit_obstacle': False, 'reached_safe': False, 'reached_meditation': False}
        
        if self.agent_pos in self.obstacles:
            reward = -5
            info['hit_obstacle'] = True
            self.agent_pos = prev_pos
        elif self.agent_pos in self.safe_zones:
            reward = 15
            info['reached_safe'] = True
        elif self.agent_pos == self.meditation_zone:
            reward = 50
            info['reached_meditation'] = True
        
        return np.array(self.agent_pos, dtype=np.int32), reward, done, truncated, info
    
    def render(self):
        self.screen.fill((255, 255, 255))
        
        for x in range(0, SCREEN_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, SCREEN_SIZE))
        for y in range(0, SCREEN_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (SCREEN_SIZE, y))
        
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (255, 0, 0), 
                            (obs[0]*CELL_SIZE, obs[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        for safe in self.safe_zones:
            pygame.draw.rect(self.screen, (255, 255, 0), 
                            (safe[0]*CELL_SIZE, safe[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        if self.meditation_zone:
            pygame.draw.rect(self.screen, (0, 255, 0), 
                            (self.meditation_zone[0]*CELL_SIZE, 
                             self.meditation_zone[1]*CELL_SIZE, 
                             CELL_SIZE, CELL_SIZE))
        
        pygame.draw.circle(self.screen, (0, 0, 255), 
                          (self.agent_pos[0]*CELL_SIZE + CELL_SIZE//2, 
                           self.agent_pos[1]*CELL_SIZE + CELL_SIZE//2), 
                          CELL_SIZE//3)
        
        text = self.font.render(f"Agent: {self.agent_pos}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))
        
        time_left = max(0, ZONE_MOVE_INTERVAL - (time.time() - self.last_zone_move_time))
        timer_text = self.font.render(f"Zones move in: {time_left:.1f}s", True, (0, 0, 0))
        self.screen.blit(timer_text, (10, 30))
        
        pygame.display.flip()
        self.clock.tick(10)
    
    def close(self):
        pygame.quit()

class StressReliefSimulator:
    def __init__(self):
        self.model_path = "models/dqn_models/dqn_stress_relief_final.zip"
        self.algorithm = MODEL_TYPE
        
        self.base_env = GridEnvironment()
        self.env = DummyVecEnv([lambda: Monitor(self.base_env)])
        
        if USE_MODEL and os.path.exists(self.model_path):
            print(f"Loading {self.algorithm.upper()} model...")
            self.model = DQN.load(self.model_path, env=self.env)
        else:
            print("Running with random actions")
            self.model = None
        
        self.total_reward = 0.0
        self.episode_count = 0
        self.steps = 0
        self.episode_rewards = []
        self.obstacle_hits = 0
        self.safe_zone_visits = 0
        self.meditation_successes = 0
        self.last_rest_time = 0
        self.resting = False

    def run_episode(self):
        obs = self.env.reset()
        done = [False]
        episode_reward = 0.0
        
        # Check if we should rest before starting this episode
        if self.episode_count > 0 and self.episode_count % REST_INTERVAL_EPISODES == 0:
            self._perform_rest()
        
        while not done[0]:
            if self.model:
                action, _ = self.model.predict(obs, deterministic=False)
            else:
                action = np.array([random.randint(0, 3)])
            
            obs, reward, done, info = self.env.step(action)
            
            if info[0].get('hit_obstacle', False):
                self.obstacle_hits += 1
            if info[0].get('reached_safe', False):
                self.safe_zone_visits += 1
            if info[0].get('reached_meditation', False):
                self.meditation_successes += 1
            
            episode_reward += float(reward[0])
            self.steps += 1
            
            self.base_env.render()
            
            for event in pygame.event.get():
                if event.type == QUIT:
                    return False
        
        self.total_reward += episode_reward
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        
        print(f"Episode {self.episode_count}: Reward {episode_reward:.1f}")
        return True

    def _perform_rest(self):
        print(f"\nAgent is resting in meditation zone for {REST_DURATION} seconds...")
        start_time = time.time()
        
        # Visual feedback during rest
        while time.time() - start_time < REST_DURATION:
            # Show resting message
            self.base_env.screen.fill((200, 230, 200))  # Light green background
            rest_text = self.base_env.font.render(
                f"Resting... {REST_DURATION - (time.time() - start_time):.1f}s remaining", 
                True, (0, 100, 0))
            self.base_env.screen.blit(rest_text, 
                                    (SCREEN_SIZE//2 - 100, SCREEN_SIZE//2 - 10))
            pygame.display.flip()
            
            # Check for quit event
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return False
            
            time.sleep(0.1)
        
        print("Agent finished resting and continues exploring.")
        return True

    def run(self):
        print("\n=== Mental AI Grid Environment ===")
        print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
        print("Colors: Agent (Blue), Obstacles (Red), Safe Zones (Yellow), Meditation (Green)")
        print(f"Zones move every {ZONE_MOVE_INTERVAL} seconds")
        print(f"Agent will rest every {REST_INTERVAL_EPISODES} episodes for {REST_DURATION} seconds")
        
        for _ in range(NUM_EPISODES):
            if not self.run_episode():
                break
        
        print("\n=== Results ===")
        print(f"Episodes: {self.episode_count}")
        print(f"Steps: {self.steps}")
        print(f"Obstacle Hits: {self.obstacle_hits}")
        print(f"Safe Zone Visits: {self.safe_zone_visits}")
        print(f"Meditation Successes: {self.meditation_successes}")
        print(f"Average Reward: {np.mean(self.episode_rewards):.1f}")
        
        self.base_env.close()

if __name__ == "__main__":
    simulator = StressReliefSimulator()
    simulator.run()
