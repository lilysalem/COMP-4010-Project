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
        self.window_size = (800, 600)
        self.hex_size = 600 // (2 * max(self.max_q, self.max_r, self.max_s) + 1)

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
    
    # returns a 6-tuple of pixel coordinates for the vertices of the hexagon
    def _hex_to_coord_set(self, coord: HexPos):
        width = self.window_size[0]
        height = self.window_size[1]

        q, r, s = coord.q, coord.r, coord.s
        assert q + r + s == 0, "q + r + s must be 0"

        x, y = q, r

        center_x = self.hex_size * np.sqrt(3) * (x + y / 2) + width // 2
        center_y = self.hex_size * 3/2 * y + height // 2

        points = []
        for i in range(6):
            angle = np.pi / 3 * i + np.pi / 6  # 30 degrees offset for pointy top
            px = center_x + self.hex_size * np.cos(angle)
            py = center_y + self.hex_size * np.sin(angle)
            points.append((int(px), int(py)))
        
        return points
    
    # draws hex cell at given pixel coordinates
    def _draw_hex_cell(self, cell: HexCell):
        cellPoints = self._hex_to_coord_set(cell.pos)
        color = GLOBALS.COLORS[cell.terrain] if GLOBALS.COLORS[cell.terrain] else (200, 200, 200)
        pygame.draw.polygon(self.window, color, cellPoints)
    
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
        
        for cell in self.hex_map.cells.values():
            self._draw_hex_cell(cell)
        
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