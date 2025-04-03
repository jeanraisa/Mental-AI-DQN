import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

class StressReliefEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None, zone_move_interval=100, static_obstacles=None, obstacle_penalty=-5):
        super().__init__()
        self.render_mode = render_mode
        self.zone_move_interval = zone_move_interval
        self.obstacle_penalty = obstacle_penalty
        
        # Environment dimensions
        self.width, self.height = 800, 600
        self.agent_size = 20
        self.zone_size = 40
        self.obstacle_size = 30
        
        # Observation space (normalized positions and distances)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(6,),
            dtype=np.float32
        )
        
        # Action space (4 discrete actions: up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Initialize positions
        self.agent_pos = None
        self.safe_zone_pos = None
        self.meditation_zone_pos = None
        self.obstacles = static_obstacles if static_obstacles else []
        
        # Movement tracking
        self.steps_since_last_move = 0
        self.current_step = 0
        self.max_steps = 1000  # Episode timeout
        
        # Rendering
        self.window = None
        self.clock = None
        self.canvas = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize agent position (center)
        self.agent_pos = np.array([self.width//2, self.height//2], dtype=np.float32)
        
        # Randomize initial zone positions
        self._randomize_zone_positions()
        
        self.steps_since_last_move = 0
        self.current_step = 0
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_observation(), {}
    
    def _randomize_zone_positions(self):
        # Generate valid positions that don't overlap
        while True:
            self.safe_zone_pos = self.np_random.integers(
                [self.zone_size, self.zone_size],
                [self.width - self.zone_size, self.height - self.zone_size],
                dtype=np.int32
            )
            self.meditation_zone_pos = self.np_random.integers(
                [self.zone_size, self.zone_size],
                [self.width - self.zone_size, self.height - self.zone_size],
                dtype=np.int32
            )
            
            # Ensure zones don't overlap and are not too close to obstacles
            if not self._check_collision(self.safe_zone_pos, self.meditation_zone_pos, self.zone_size, self.zone_size):
                valid = True
                for obstacle in self.obstacles:
                    if self._check_collision(self.safe_zone_pos, obstacle, self.zone_size, self.obstacle_size) or \
                       self._check_collision(self.meditation_zone_pos, obstacle, self.zone_size, self.obstacle_size):
                        valid = False
                        break
                if valid:
                    break

    def step(self, action):
        self.current_step += 1
        
        # Move agent (5 pixels per step)
        direction = {
            0: np.array([0, -5]),   # up
            1: np.array([0, 5]),    # down
            2: np.array([-5, 0]),   # left
            3: np.array([5, 0])     # right
        }[action]
        
        new_pos = self.agent_pos + direction
        new_pos = np.clip(new_pos, [0, 0], [self.width-self.agent_size, self.height-self.agent_size])
        
        # Calculate distances for reward shaping
        prev_med_dist = np.linalg.norm(self.agent_pos - self.meditation_zone_pos)
        prev_safe_dist = np.linalg.norm(self.agent_pos - self.safe_zone_pos)
        
        # Check obstacle collisions
        hit_obstacle = any(
            self._check_collision(new_pos, obstacle, self.agent_size, self.obstacle_size)
            for obstacle in self.obstacles
        )
        
        # Update position
        self.agent_pos = new_pos
        
        # Calculate new distances
        new_med_dist = np.linalg.norm(self.agent_pos - self.meditation_zone_pos)
        new_safe_dist = np.linalg.norm(self.agent_pos - self.safe_zone_pos)
        
        # Base reward calculation
        reward = -0.05  # Small time penalty
        
        # Distance-based shaping rewards
        reward += (prev_med_dist - new_med_dist) * 0.2  # Reward for moving closer to meditation zone
        reward += (prev_safe_dist - new_safe_dist) * 0.1  # Smaller reward for moving closer to safe zone
        
        # Check zone visits
        reached_safe = self._check_collision(self.agent_pos, self.safe_zone_pos, self.agent_size, self.zone_size)
        reached_meditation = self._check_collision(self.agent_pos, self.meditation_zone_pos, self.agent_size, self.zone_size)
        
        if hit_obstacle:
            reward += self.obstacle_penalty
            
        if reached_safe:
            reward += 15  # Increased safe zone reward
            
        if reached_meditation:
            reward += 50  # Large success reward
            terminated = True
        else:
            terminated = False
            
        # Timeout termination
        truncated = self.current_step >= self.max_steps
        
        # Zone movement
        self.steps_since_last_move += 1
        if self.steps_since_last_move >= self.zone_move_interval:
            self._randomize_zone_positions()
            self.steps_since_last_move = 0
            
        if self.render_mode == "human":
            self.render()
            
        return (
            self._get_observation(),
            reward,
            terminated or truncated,
            truncated,
            {
                'hit_obstacle': hit_obstacle,
                'reached_safe': reached_safe,
                'reached_meditation': reached_meditation
            }
        )
    
    def _get_observation(self):
        # Normalized observations with distance metrics
        return np.array([
            # Normalized positions
            self.agent_pos[0] / self.width,
            self.agent_pos[1] / self.height,
            self.meditation_zone_pos[0] / self.width,
            self.meditation_zone_pos[1] / self.height,
            # Relative distances
            np.linalg.norm(self.agent_pos - self.meditation_zone_pos) / np.sqrt(self.width**2 + self.height**2),
            np.linalg.norm(self.agent_pos - self.safe_zone_pos) / np.sqrt(self.width**2 + self.height**2)
        ], dtype=np.float32)
    
    def _check_collision(self, pos1, pos2, size1, size2):
        return (abs(pos1[0] - pos2[0]) < (size1 + size2) and 
                abs(pos1[1] - pos2[1]) < (size1 + size2))
    
    def render(self):
        if self.render_mode is None:
            raise gym.error.Error(
                "You must specify a render_mode when initializing the environment"
            )
        return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Stress Relief Environment")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        self.canvas = pygame.Surface((self.width, self.height))
        self.canvas.fill((255, 255, 255))
        
        # Draw obstacles (static)
        for obstacle in self.obstacles:
            pygame.draw.rect(
                self.canvas, (255, 0, 0),
                pygame.Rect(obstacle[0], obstacle[1], self.obstacle_size, self.obstacle_size)
            )
            
        # Draw safe zone (green)
        pygame.draw.rect(
            self.canvas, (0, 255, 0),
            pygame.Rect(
                self.safe_zone_pos[0], self.safe_zone_pos[1],
                self.zone_size, self.zone_size
            )
        )
        
        # Draw meditation zone (blue)
        pygame.draw.rect(
            self.canvas, (0, 0, 255),
            pygame.Rect(
                self.meditation_zone_pos[0], self.meditation_zone_pos[1],
                self.zone_size, self.zone_size
            )
        )
        
        # Draw agent (black)
        pygame.draw.rect(
            self.canvas, (0, 0, 0),
            pygame.Rect(
                self.agent_pos[0], self.agent_pos[1],
                self.agent_size, self.agent_size
            )
        )
        
        if self.render_mode == "human":
            self.window.blit(self.canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
