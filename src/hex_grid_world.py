"""
Main environment.
The world manager.
Owns the grid and the colony list.
Sets up variables and all that.
Runs the simulation.

Includes main for testing for now.

Files this env version needs:
-hex_grid_world.py (this one)
-hex_grid.py
-ants.py
-animator.py
-worlds.py

New files to create:
-Reinforcement learning file
-Testing file

"""

# Imports
import gymnasium as gym
from hex_grid import HexGrid
import ants
import window_animator
import worlds
from random import randint
from time import sleep # For the animation


# World manager
class HexGridWorld(gym.Env):
    train: bool # Train AI
    worldType: int # World style, 0 == random map, >=1 == preset map
    xR: int # X range (world dimension, going up)
    yR: int # Y range (world dimension, going right and down)
    zR: int # Z range (world dimension, going left and down)
    stepCount: int = 0 # Track number of steps taken
    grid: HexGrid # The world
    gridMemory: list = None # Store random map for resets
    colony: list[ants.Ant] = [] # The ants
    animate: bool = False # Toggle Pygame rendering (unnecessary while training)
    animator: window_animator.Animator = None # The Pygame display handler
    
    # Initialize
    def __init__(self, train: bool, worldType: int, x: int = None, y: int = None, z:int = None, animate: int = False, windowSize: tuple[int, int] = (1250, 750)):
        # Setup
        self.train = train
        self.worldType = worldType
        # Preset will override these, random will fill in the gaps
        self.xR = x
        self.yR = y
        self.zR = z
        # Generate world
        self.buildWorld()
        # Set up animation
        if animate:
            self.animate = True
            self.animator = window_animator.Animator(self.xR, self.yR, self.zR, windowSize)
            self.render()
            sleep(1)
        # Go!
        print(self.xR, " ", self.yR, " ", self.zR)
        print("Started")

    # Reset
    def reset(self):
        # Wipe and regenerate world
        self.grid = None
        self.colony = []
        self.buildWorld()
        # Keep going!
        print("Reset")
        # Pass in animation window
        if self.animate:
            self.render()
            sleep(1)
    
    # Run simulation step
    def step(self, action: int) -> tuple[tuple[bool, str, str, str] | None, int | None, bool, bool, str | None]:
        s = None
        a = None
        r = None
        s_ = None
        info = None
        if self.train:
            # Only one worker is trained at a time (unless we figure out how to do multiple?)
            s, a, r, s_ = self.colony[1].act(action = action)
            # (Other training crap here)
        else:
            # All ants in the colony take an action each step
            # Turn order is colony list order
            actCycle = 0
            while actCycle < len(self.colony):
                self.colony[actCycle].act()
                actCycle += 1
            # Post-action stuff
            # Reduce the strength of trails over time
            self.grid.fadeAllTrails()
        # Kill worker ants that have reached the ends of their lives
        # Lifespan is 1000 steps, this is changeable
        # Queen is immortal and unaffected (unless we change that!)
        for ant in self.colony:
            if ant.age >= 1000:
                ant.die()
                self.colony.remove(ant)
                del ant
        # Increment step
        self.stepCount += 1
        # Animate
        if self.animate:
            self.render()
            sleep(0.01) # Animation speed, toggle/change this as desired
        # End simulation if 1 food retrieved while training or if colony died off
        terminated = False
        truncated = False
        if self.train and self.colony[0].food > 0:
            terminated = True
            print("Terminated")
        elif len(self.colony) == 1:
            truncated = True
            print("Truncated")
        return s_, r, terminated, truncated, info # Continue
    
    # Update animation
    def render(self):
        self.animator.drawFullGrid(self.grid)
        for ant in self.colony: # Grid knows where the ants are but knows nothing about them so gotta do them separately
            if ant == self.colony[0]: # Queen
                self.animator.drawCell(self.grid, (ant.x,ant.y,ant.z), antDir = ant.dir)
            else: # Workers
                self.animator.drawCell(self.grid, (ant.x,ant.y,ant.z), antDir = ant.dir, antHasFood = ant.hasFood)
        self.animator.updateWindow()

    # Close - gotta add this
    def close(self):
        return
    
    # Generate the correct world type
    # To add more: build its function and add it to this list
    def buildWorld(self):
        if self.worldType == 1:
            worlds.presetWorld1(self)
        # Add more here
        else: # 0 plus catch-all
            worlds.randomWorld(self)
        

# Test script
# This should be moved to a new file once we get more detailed tests going for training
if __name__ == "__main__":
    print("TEST")
    # Demo world
    world = HexGridWorld(False, 0, animate = True)
    # Run!
    for i in range(3):
        terminated = False
        truncated = False
        while not (terminated or truncated):
            s_, r, terminated, truncated, info = world.step(randint(0, 4))
        world.reset()
    # Display briefly when done
    #sleep(5)
    print("DONE")
