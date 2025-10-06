import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional
import src.globals as GLOBALS
from src.hex_grid_helpers import HexPos, HexCell, HexMap


class HexGridWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        grid_radius: int = 10,
        render_mode: Optional[str] = None,
        max_steps: int = 100,
        world_seed: Optional[int] = None
    ):
        self.max_q = grid_radius
        self.max_r = grid_radius
        self.max_s = grid_radius
        self.hex_map = HexMap(grid_radius, grid_radius, grid_radius, seed=str(world_seed))
        
    def _is_valid_pos(self, pos):
        """Check if a position is within grid boundaries and not an obstacle."""
        q, r = pos
        # Check grid boundaries
        if q < self.min_q or q > self.max_q or r < self.min_r or r > self.max_r:
            return False
        # Check if position is an obstacle
        if pos in self.obstacles:
            return False
        # Check if the position satisfies the third coordinate constraint for hexagonal grid
        # In axial coordinates, we implicitly have s = -q-r, and need to check if |s| <= grid_size
        s = -q - r
        if abs(s) > self.grid_size:
            return False
        return True
        
    def _get_random_valid_pos(self):
        """Get a random valid position on the grid."""
        while True:
            q = np.random.randint(self.min_q, self.max_q + 1)
            r = np.random.randint(self.min_r, self.max_r + 1)
            pos = (q, r)
            if self._is_valid_pos(pos):
                return pos
    
    def _generate_obstacles(self, num_obstacles=None):
        """Generate random obstacles on the grid."""
        if num_obstacles is None:
            # Default: add obstacles for about 20% of the grid
            total_cells = (2 * self.grid_size + 1) ** 2
            num_obstacles = int(0.2 * total_cells)
        
        self.obstacles = []
        for _ in range(num_obstacles):
            # Make sure obstacle is not at agent or target position
            while True:
                pos = self._get_random_valid_pos()
                if pos != self.agent_pos and pos != self.target_pos and pos not in self.obstacles:
                    self.obstacles.append(pos)
                    break
                    
    def _calculate_reward(self):
        """Calculate reward based on current state."""
        # Reward for reaching target
        if self.agent_pos == self.target_pos:
            return 10.0
            
        # Small penalty for each step to encourage shortest path
        return -0.1
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Seed for random number generator
            options: Additional options (not used currently)
            
        Returns:
            observation: The initial observation
            info: Additional information
        """
        # Initialize RNG
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Set agent position at center
        self.agent_pos = (0, 0)
        
        # Set random target position (not the same as agent)
        while True:
            self.target_pos = self._get_random_valid_pos()
            if self.target_pos != self.agent_pos:
                break
        
        # Generate obstacles
        self._generate_obstacles()
        
        # Initialize renderer
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: An action to take (0-5)
            
        Returns:
            observation: The new observation
            reward: The reward for taking the action
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated (e.g., due to max steps)
            info: Additional information
        """
        # Increment step counter
        self.step_count += 1
        
        # Get the direction vector
        direction = self.DIRECTIONS[action]
        
        # Calculate new position
        new_q = self.agent_pos[0] + direction[0]
        new_r = self.agent_pos[1] + direction[1]
        new_pos = (new_q, new_r)
        
        # Move agent if valid position
        if self._is_valid_pos(new_pos):
            self.agent_pos = new_pos
            
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if terminated
        terminated = self.agent_pos == self.target_pos
        
        # Check if truncated due to max steps
        truncated = self.step_count >= self.max_steps
        
        # Update rendering
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _get_obs(self):
        """Get current observation."""
        return np.array([self.agent_pos[0], self.agent_pos[1]], dtype=np.int32)
    
    def _get_info(self):
        """Get additional information."""
        # Calculate Manhattan distance to target in hex grid
        q_dist = abs(self.agent_pos[0] - self.target_pos[0])
        r_dist = abs(self.agent_pos[1] - self.target_pos[1])
        s_dist = abs(-self.agent_pos[0] - self.agent_pos[1] - (-self.target_pos[0] - self.target_pos[1]))
        # Hexagonal distance is max of the three coordinates
        hex_distance = max(q_dist, r_dist, s_dist)
        
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