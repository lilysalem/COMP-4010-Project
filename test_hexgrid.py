#!/usr/bin/env python3
"""
Testbed for the hexagonal grid world environment.
This script demonstrates how to use the HexGridWorld environment.
"""

import gymnasium as gym
import numpy as np
import time
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.environment import HexGridWorld

def test_manual_agent():
    """Test manual control in the hexagonal grid world."""
    import pygame
    
    # Create the environment
    env = HexGridWorld(grid_size=5, render_mode="human")
    observation, info = env.reset(seed=42)
    
    print("Starting test with manual control...")
    print("Use keyboard for control:")
    print("  Q/W: Move Northeast/Northwest")
    print("  A/S: Move West/East")
    print("  Z/X: Move Southwest/Southeast")
    print("  ESC: Exit")
    
    total_reward = 0
    num_steps = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                action = None
                
                # Map keys to actions
                if event.key == pygame.K_s:
                    action = 0  # East
                elif event.key == pygame.K_q:
                    action = 1  # Northeast
                elif event.key == pygame.K_w:
                    action = 2  # Northwest
                elif event.key == pygame.K_a:
                    action = 3  # West
                elif event.key == pygame.K_z:
                    action = 4  # Southwest
                elif event.key == pygame.K_x:
                    action = 5  # Southeast
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    
                if action is not None:
                    observation, reward, terminated, truncated, info = env.step(action)
                    
                    total_reward += reward
                    num_steps += 1
                    
                    print(f"Step {num_steps}: Action={action}, Pos={observation}, Reward={reward}, Info={info}")
                    
                    if terminated or truncated:
                        running = False
        
        # Cap the frame rate
        pygame.time.delay(50)
    
    # Close the environment
    env.close()
    
    print(f"Episode ended after {num_steps} steps with total reward: {total_reward}")
    if terminated:
        print("Agent reached the target!")
    elif truncated and num_steps > 0:
        print("Episode truncated due to maximum steps.")

def register_env():
    """Register our custom environment with Gymnasium."""
    from gymnasium.envs.registration import register
    
    register(
        id='HexGridWorld-v0',
        entry_point='src.hexgrid_env:HexGridWorld',
        max_episode_steps=100,
    )
    
    print("HexGridWorld environment registered successfully.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the hexagonal grid world environment')
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'manual', 'register'],
                        help='Test mode: random agent, manual control, or register environment')
    
    args = parser.parse_args()
    
    if args.mode == 'random':
        test_random_agent()
    elif args.mode == 'manual':
        test_manual_agent()
    elif args.mode == 'register':
        register_env()
        
        # Test the registered environment
        env = gym.make('HexGridWorld-v0', render_mode="human")
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())
        env.close()