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
    def step(self, action: int | None) -> tuple[tuple[bool, str, str, str] | None, int | None, bool, bool, str | None]:
        s = None
        a = None
        r = None
        s_ = None
        info = None
        if self.train:
            if len(self.colony) > 1 and self.colony[1].q_agent is None:
                from q_learning import QLearningAgent
                self.colony[1].q_agent = QLearningAgent()

            if len(self.colony) > 1:
                s, a, r, s_ = self.colony[1].act(action = action)
            else:
                s, a, r, s_ = None, None, 0, None
        else:
            r = 0  # Initialize reward for evaluation mode
            actCycle = 0
            while actCycle < len(self.colony):
                # Only call act() on workers if they have a Q-agent
                # Queen (index 0) doesn't need Q-agent, Worker (index 1+) does
                if actCycle == 0: # The queen doesn't need a Q-agent, it just acts normally
                    self.colony[actCycle].act()
                elif actCycle > 0 and self.colony[actCycle].q_agent is not None: # The worker needs a Q-agent to act
                    # Capture reward from worker's act() call for evaluation
                    s, a, r_worker, s_ = self.colony[actCycle].act()
                    # Accumulate reward from worker
                    if r_worker is not None:
                        r += r_worker
                actCycle += 1
            self.grid.fadeAllTrails()

        for ant in self.colony:
            if ant.age >= 5000:
                ant.die()
                self.colony.remove(ant)
                del ant

        self.stepCount += 1

        if self.animate:
            self.render()

        terminated = False
        truncated = False
        # Checks if the colony exists and the queen has food. We terminate based on whether queen has food, not dependent on the self.train mode.
        if len(self.colony) > 0 and self.colony[0].food > 0: #changed from self.train to self.colony because self.train = False during evaluation mode and then the episode would never terminate
            terminated = True
            if self.train:
                print("Terminated")
        elif len(self.colony) == 1:
            truncated = True
            if self.train:
                print("Truncated")
        return s_, r, terminated, truncated, info
    
    def render(self):
        self.animator.drawFullGrid(self.grid)
        for ant in self.colony:
            if ant == self.colony[0] or type(ant) == ants.Queen:
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
