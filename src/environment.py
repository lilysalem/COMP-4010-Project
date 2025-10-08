from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional
import src.globals as GLOBALS
from src.hex_grid_helpers import HexPos, HexCell, HexMap

@dataclass
class HexGridWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    step_count: int
    max_steps: int
    max_q: int
    max_r: int
    max_s: int
    hex_map: HexMap
    world_seed: Optional[str] = None

    def __post_init__(self):
        self.hex_map = HexMap(self.max_q, self.max_r, self.max_s, seed=str(self.world_seed))

    def _is_valid_pos(self, pos: HexPos):
        return self.hex_map.is_pos_valid(pos)
        
    def _get_random_valid_pos(self):
        while True:
            q = np.random.randint(self.min_q, self.max_q + 1)
            r = np.random.randint(self.min_r, self.max_r + 1)
            s = -q - r
            pos = HexPos(q, r, s)
            if self._is_valid_pos(pos):
                return pos
                    
    def _calculate_reward(self):
        if self.agent_pos == self.target_pos:
            return 1.0
            
        return -0.01
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        
        # Set agent position at center
        self.agent_pos = HexPos(0, 0, 0)

        while True:
            self.target_pos = self._get_random_valid_pos()
            if self.target_pos != self.agent_pos:
                break
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        self.step_count += 1
        
        direction = GLOBALS.DIRECTIONS[action]
        
        new_pos = HexPos(self.agent_pos.q + direction.q, 
                        self.agent_pos.r + direction.s, 
                        self.agent_pos.s + direction.s)

        if self._is_valid_pos(new_pos):
            self.agent_pos = new_pos

        reward = self._calculate_reward()

        terminated = self.agent_pos == self.target_pos
        
        truncated = self.step_count >= self.max_steps
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.int32)
    
    def _get_info(self):
        """Get additional information."""
        agent = self.agent_pos
        target = self.target_pos
        q_dist = abs(agent.q - target.q)
        r_dist = abs(agent.r - target.r)
        s_dist = abs(agent.s - target.s)

        hex_distance = (q_dist + r_dist + s_dist) // 2
        
        return {
            "distance_to_target": hex_distance,
            "steps_taken": self.step_count
        }
    
    def _hex_to_pixel(self, q, r):
        """Convert hexagonal coordinates to pixel coordinates for rendering."""
        # Constants for rendering
        size = self.hex_size
        width = size * 2
        height = np.sqrt(3) * size
        
        # Calculate pixel coordinates
        x = width * (q + r/2)
        y = height * r
        
        # Center the grid on the screen
        x += self.window_size[0] / 2
        y += self.window_size[1] / 2
        
        return int(x), int(y)
    
    def _draw_hexagon(self, surface, color, center_x, center_y):
        """Draw a hexagon on the surface."""
        size = self.hex_size
        points = []
        
        for i in range(6):
            angle = np.pi / 3 * i
            x = center_x + size * np.cos(angle)
            y = center_y + size * np.sin(angle)
            points.append((int(x), int(y)))
            
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (0, 0, 0), points, 2)  # Black outline
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
            
        return self._render_frame()
    
    def _render_frame(self):
        """Render a frame."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))  # White background
        
        # Draw all valid hexagons
        for q in range(self.min_q, self.max_q + 1):
            for r in range(self.min_r, self.max_r + 1):
                pos = (q, r)
                if self._is_valid_pos(pos):
                    pixel_x, pixel_y = self._hex_to_pixel(q, r)
                    
                    # Different colors for different cell types
                    color = (200, 200, 200)  # Default gray
                    
                    if pos in self.obstacles:
                        color = (100, 100, 100)  # Darker gray for obstacles
                    if pos == self.target_pos:
                        color = (0, 255, 0)     # Green for target
                    if pos == self.agent_pos:
                        color = (255, 0, 0)     # Red for agent
                        
                    self._draw_hexagon(canvas, color, pixel_x, pixel_y)
        
        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Close the environment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None