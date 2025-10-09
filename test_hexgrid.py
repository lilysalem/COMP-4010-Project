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
    # Initialize the environment
    env = HexGridWorld(max_q=5, max_r=5, max_s=5, max_steps=100, render_mode="human")
    
    # Reset the environment
    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Additional Info:", info)

    print("Use the following keys to move the agent:")
    print("0: East, 1: Northeast, 2: Northwest, 3: West, 4: Southwest, 5: Southeast")
    print("Press 'q' to quit.")

    while True:
        # Render the environment
        env.render()

        # Get user input for action
        action = input("Enter action (0-5) or 'q' to quit: ")

        if action.lower() == 'q':
            print("Exiting manual control.")
            break

        if action.isdigit() and int(action) in range(6):
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)

            print("Observation:", obs)
            print("Reward:", reward)
            print("Terminated:", terminated)
            print("Truncated:", truncated)
            print("Info:", info)

            if terminated or truncated:
                print("Episode finished. Resetting environment.")
                obs, info = env.reset()
        else:
            print("Invalid action. Please enter a number between 0 and 5, or 'q' to quit.")

    env.close()


if __name__ == "__main__":
    # import argparse
    
    # parser = argparse.ArgumentParser(description='Test the hexagonal grid world environment')
    # parser.add_argument('--mode', type=str, default='random', choices=['random', 'manual', 'register'],
    #                     help='Test mode: random agent, manual control, or register environment')
    
    # args = parser.parse_args()
    
    test_manual_agent()