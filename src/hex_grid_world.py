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
            if len(self.colony) > 1 and self.colony[1].q_agent is None:
                self.colony[1].q_agent = ants.QLearningAgent()

            if len(self.colony) > 1:
                s, a, r, s_ = self.colony[1].act(action = action)
            else:
                s, a, r, s_ = None, None, 0, None
        else:
            actCycle = 0
            while actCycle < len(self.colony):
                self.colony[actCycle].act()
                actCycle += 1
            self.grid.fadeAllTrails()

        for ant in self.colony:
            if ant.age >= 1000:
                ant.die()
                self.colony.remove(ant)
                del ant

        self.stepCount += 1

        if self.animate:
            self.render()

        terminated = False
        truncated = False
        if self.train and self.colony[0].food > 0:
            terminated = True
            print("Terminated")
        elif len(self.colony) == 1:
            truncated = True
            print("Truncated")
        return s_, r, terminated, truncated, info
    
    def render(self):
        self.animator.drawFullGrid(self.grid)
        for ant in self.colony:
            if ant == self.colony[0]:
                self.animator.drawCell(self.grid, (ant.x,ant.y,ant.z), antDir = ant.dir)
            else:
                self.animator.drawCell(self.grid, (ant.x,ant.y,ant.z), antDir = ant.dir, antHasFood = ant.hasFood)
        self.animator.updateWindow()

    def close(self):
        return

    def buildWorld(self):
        if self.worldType == 1:
            worlds.presetWorld1(self)
        else:
            worlds.randomWorld(self)
        

if __name__ == "__main__":
    print("TEST")
    world = HexGridWorld(False, 0, animate = True)
    for i in range(3):
        terminated = False
        truncated = False
        while not (terminated or truncated):
            s_, r, terminated, truncated, info = world.step(randint(0, 4))
        world.reset()
    print("DONE")
