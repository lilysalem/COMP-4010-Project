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

New files to create:
-Reinforcement learning file for state/action/reward calculations and storage
(don't want it to clog up the ant file, it can send its info over to another class)
-Testing file for more specific env runs during training

"""

# Imports
import gymnasium as gym
import hex_grid
import ants
import random # from random import randint? I haven't used anything but that function
from time import sleep # For the animation
#import numpy as np # Unused while we don't have the reinforcement math, it'll go in that file


# World manager
class HexGridWorld(gym.Env):
    xR: int # X range (world dimension, going up)
    yR: int # Y range (world dimension, going right and down)
    zR: int # Z range (world dimension, going left and down)
    stepMax: int # Stop when this step count is reached (optional)
    stepCount: int = 0 # Track number of steps taken
    grid: hex_grid.HexGrid # The world
    colony: list = [] # The ants
    animate: bool = False # Toggle Pygame rendering (unnecessary while training)
    
    # Initialize
    def __init__(self, x, y = None, z = None, stepMax = 0, animate = False, windowSize = (1250, 750)):
        # World dimensions
        # Only one is required, others default to that one if not given
        # So if only that one is given it makes a hexagon
        self.xR = x
        if y != None:
            self.yR = y
        else:
            self.yR = x
        if z != None:
            self.zR = z
        else:
            self.zR = x
        
        # Set step cutoff
        self.stepMax = stepMax

        # Initialize grid environment
        self.grid = hex_grid.HexGrid(self.xR, self.yR, self.zR)

        # Animation activation
        # The grid handles the Pygame rendering, but not automatically
        # It depends on external calls from this class
        if animate:
            self.animate = True
            self.grid.createWindow(windowSize)
        
        # Populate world with objects, food, and the queen
        self.generateWorld()

        # Go!
        print("Started")

    # Build the world before running the simulation
    # This is extremely flexible as long as the queen gets placed
    # Generation is random for now but we could move that to a separate function
    # We could add world preset styles for training based on an input param
    # Possibly: randomWorld() and presetWorld(ID), called from this function
    def generateWorld(self):

        # Random world generation
        maxRockSize = int(min(self.xR, self.yR, self.zR) * 0.15)
        # Three nested loops, each covering the sector with the two used axes plus one of those axes
        for i in range(self.xR - 1):
            for ii in range(1, self.yR - 1):
                # Randomly determine what object to place
                gen = random.randint(0,29)
                if gen == 0: # Food
                    self.grid.setCell((i,ii,0), "F")
                if gen == 1: # Obstacle
                    self.grid.setCell((i,ii,0), "O")
                if gen == 2: # Obstacle but larger cluster
                    self.buildCluster((i,ii,0), random.randint(1, maxRockSize), "O")
        for i in range(self.yR - 1):
            for ii in range(1, self.zR - 1):
                gen = random.randint(0,29)
                if gen == 0:
                    self.grid.setCell((0,i,ii), "F")
                if gen == 1:
                    self.grid.setCell((0,i,ii), "O")
                if gen == 2:
                    self.buildCluster((0,i,ii), random.randint(1, maxRockSize), "O")
        for i in range(self.zR - 1):
            for ii in range(1, self.xR - 1):
                gen = random.randint(0,29)
                if gen == 0:
                    self.grid.setCell((ii,0,i), "F")
                if gen == 1:
                    self.grid.setCell((ii,0,i), "O")
                if gen == 2:
                    self.buildCluster((ii,0,i), random.randint(1, maxRockSize), "O")

        # Large food clusters
        # Three, one close to the end of each axis, size also dependent on that axis
        pileX = int(self.xR * 0.75)
        sizeX = int((self.xR - pileX) * 0.5)
        pileY = int(self.yR * 0.75)
        sizeY = int((self.yR - pileY) * 0.5)
        pileZ = int(self.zR * 0.75)
        sizeZ = int((self.zR - pileZ) * 0.5)
        self.buildCluster((pileX,0,0), sizeX, "F")
        self.buildCluster((0,pileY,0), sizeY, "F")
        self.buildCluster((0,0,pileZ), sizeZ, "F")

        # Clear space for colony
        # This is also modifiable but they need at least some space to move around the queen
        # If they don't have it they get into a traffic jam
        space = 3 # Number of free cells around the queen on each axis
        self.buildCluster((0,0,0), space + 1, "E")
        
        # Create queen - ready to run!
        # Queen is always 0,0,0 for now, this is again changeable if we add params
        self.colony.append(ants.Queen(self.grid))
        self.colony[0].colony = self.colony # Queen gets colony access for spawning
        self.grid.setCell((0,0,0), "Q") # Mark on grid

        # Animate
        if self.animate:
            self.grid.drawFullGrid()
            self.grid.render()
            sleep(1)
    
    # Helper for filling hexagonal clusters of cells with a tile type
    def buildCluster(self, c, scale, cellType):
        x, y, z = c[0], c[1], c[2]
        # Centre cell
        if self.grid.isWithinGrid(c):
            self.grid.setCell(c, cellType)
        # Everything else
        for i in range(scale):
            for ii in range(1, scale):
                if self.grid.isWithinGrid((x+i,y+ii,z)):
                    self.grid.setCell(self.grid.normalize((x+i,y+ii,z)), cellType)
                if self.grid.isWithinGrid((x,y+i,z+ii)):
                    self.grid.setCell(self.grid.normalize((x,y+i,z+ii)), cellType)
                if self.grid.isWithinGrid((x+ii,y,z+i)):
                    self.grid.setCell(self.grid.normalize((x+ii,y,z+i)), cellType)

    # Run simulation
    def step(self):
        #print(self.stepCount)

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
            self.grid.drawFullGrid()
            for ant in self.colony: # Grid knows where the ants are but knows nothing about them - might give it colony access later?
                if ant == self.colony[0]: # Queen
                    self.grid.drawCell((ant.x,ant.y,ant.z), antDir = ant.dir)
                else: # Workers
                    self.grid.drawCell((ant.x,ant.y,ant.z), antDir = ant.dir, antHasFood = ant.hasFood)
            self.grid.render()
            sleep(0.01) # Animation speed, toggle/change this as desired
        
        # End simulation
        # Step count reaches max or colony died off
        if (self.stepMax >= 1 and self.stepCount >= self.stepMax) or len(self.colony) == 1:
            # Stop!
            print("Ended")
            #self.grid.closeWindow()
            #exit()
            return False # Stop
        return True # Continue


# Test script.
# This should be moved to a new file once we get more detailed tests going for training.
if __name__ == "__main__":
    print("TEST")
    # Set up a demo world
    world = HexGridWorld(25, y = 50, z = 75, animate = True) #, windowSize = (750,500)
    # Run!
    run = True
    while run:
        run = world.step()
    # Display briefly when done
    sleep(5)
