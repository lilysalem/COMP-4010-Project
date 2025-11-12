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
from random import randint
from time import sleep # For the animation
#import numpy as np # Unused while we don't have the reinforcement math, it'll go in that file


# World manager
class HexGridWorld(gym.Env):
    train: bool # Train AI
    worldType: int # World style, 0 == random map, >=1 == preset map
    xR: int # X range (world dimension, going up)
    yR: int # Y range (world dimension, going right and down)
    zR: int # Z range (world dimension, going left and down)
    stepCount: int = 0 # Track number of steps taken
    grid: hex_grid.HexGrid # The world
    gridMemory: list = None # Store random map for resets
    colony: list = [] # The ants
    animate: bool = False # Toggle Pygame rendering (unnecessary while training)
    
    # Initialize
    def __init__(self, train, worldType, x = None, y = None, z = None, animate = False, windowSize = (1250, 750)):
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
            self.grid.createWindow(windowSize)
            self.grid.drawFullGrid()
            self.grid.render()
            sleep(1)
        # Go!
        print(self.xR, " ", self.yR, " ", self.zR)
        print("Started")

    # Reset
    def reset(self):
        # Save animation window
        if self.animate:
            windowData = self.grid.passWindowData()
        # Wipe and regenerate world
        self.grid = None
        self.colony = []
        self.buildWorld()
        # Keep going!
        print("Reset")
        # Pass in animation window
        if self.animate:
            self.grid.loadWindow(windowData)
            self.grid.drawFullGrid()
            self.grid.render()
            sleep(1)
    
    # Generate the correct world type
    # To add more: build its function and add it to this list
    def buildWorld(self):
        if self.worldType == 1:
            self.presetWorld1()
        # Add more here
        else: # 0 plus catch-all
            self.randomWorld()
    
    # World 0 (default random)
    # Random dimensions, terrain, food
    # Stores the world on the first episode and loads it on subsequent
    def randomWorld(self):
        # First episode
        if self.gridMemory == None:
            # New random map, first episode
            if self.xR == None:
                self.xR = randint(10, 100)
            if self.yR == None:
                self.yR = randint(10, 100)
            if self.zR == None:
                self.zR = randint(10, 100)
            # Create grid
            self.grid = hex_grid.HexGrid(self.xR, self.yR, self.zR)
            # Random world generation
            maxRockSize = int(min(self.xR, self.yR, self.zR) * 0.1)
            # Three nested loops, each covering the sector with the two used axes plus one of those axes
            for i in range(self.xR):
                for ii in range(1, self.yR):
                    gen = randint(0,29) # Randomly determine what object to place
                    if gen == 0: # Food
                        self.grid.setCell((i,ii,0), "F")
                    if gen == 1: # Obstacle
                        self.grid.setCell((i,ii,0), "O")
                    if gen == 2: # Obstacle but larger cluster
                        self.buildCluster((i,ii,0), randint(1, maxRockSize), "O")
            for i in range(self.yR):
                for ii in range(1, self.zR):
                    gen = randint(0,29)
                    if gen == 0:
                        self.grid.setCell((0,i,ii), "F")
                    if gen == 1:
                        self.grid.setCell((0,i,ii), "O")
                    if gen == 2:
                        self.buildCluster((0,i,ii), randint(1, maxRockSize), "O")
            for i in range(self.zR):
                for ii in range(1, self.xR):
                    gen = randint(0,29)
                    if gen == 0:
                        self.grid.setCell((ii,0,i), "F")
                    if gen == 1:
                        self.grid.setCell((ii,0,i), "O")
                    if gen == 2:
                        self.buildCluster((ii,0,i), randint(1, maxRockSize), "O")
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
            # If they don't have any they get into a traffic jam
            space = 3 # Number of free cells around the queen on each axis
            self.buildCluster((0,0,0), space + 1, "E")
            # Store map
            self.gridMemory = ["Q"]
            for i in range(self.xR):
                for ii in range(1, self.yR):
                    self.gridMemory.append(self.grid.getCell((i,ii,0)))
            for i in range(self.yR):
                for ii in range(1, self.zR):
                    self.gridMemory.append(self.grid.getCell((0,i,ii)))
            for i in range(self.zR):
                for ii in range(1, self.xR):
                    self.gridMemory.append(self.grid.getCell((ii,0,i)))
        # Subsequent episodes
        else:
            # Load previously saved map
            self.grid = hex_grid.HexGrid(self.xR, self.yR, self.zR)
            self.grid.setCell((0,0,0), self.gridMemory[0])
            iterateCell = 1
            for i in range(self.xR):
                for ii in range(1, self.yR):
                    self.grid.setCell((i,ii,0), self.gridMemory[iterateCell])
                    iterateCell += 1
            for i in range(self.yR):
                for ii in range(1, self.zR):
                    self.grid.setCell((0,i,ii), self.gridMemory[iterateCell])
                    iterateCell += 1
            for i in range(self.zR):
                for ii in range(1, self.xR):
                    self.grid.setCell((ii,0,i), self.gridMemory[iterateCell])
                    iterateCell += 1
        # Create queen
        # Always at 0,0,0 in random world
        self.colony.append(ants.Queen(self.grid))
        self.colony[0].colony = self.colony # Queen gets colony access for spawning
        self.grid.setCell((0,0,0), "Q") # Mark on grid
        self.colony[0].food = 1 # One worker to start
        self.colony[0].act() # Spawn that worker at random adjacent location
    
    # World 1
    # Small, queen at bottom, one food at top, pre-drawn trail straight between them
    def presetWorld1(self):
        self.xR = 10
        self.yR = 10
        self.zR = 10
        self.grid = hex_grid.HexGrid(self.xR, self.yR, self.zR)
        self.grid.setCell((9,0,0), "F")
        for i in range(-8, 9):
            self.grid.setTrail(self.grid.normalize((i,0,0)), 25)
        self.colony.append(ants.Queen(self.grid, y = 9, z = 9))
        self.colony[0].colony = self.colony
        self.grid.setCell((0,9,9), "Q")
        self.colony.append(ants.Worker(self.grid, y = 8, z = 8))
        self.colony[1].queen = self.colony[0]
        self.grid.setCell((0,8,8), "W")
    
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

    # Run simulation step
    def step(self):
        #print(self.stepCount)
        if self.train:
            # Only one worker is trained at a time (unless we figure out how to do multiple?)
            self.colony[1].act()
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
            self.grid.drawFullGrid()
            for ant in self.colony: # Grid knows where the ants are but knows nothing about them - might give it colony access later?
                if ant == self.colony[0]: # Queen
                    self.grid.drawCell((ant.x,ant.y,ant.z), antDir = ant.dir)
                else: # Workers
                    self.grid.drawCell((ant.x,ant.y,ant.z), antDir = ant.dir, antHasFood = ant.hasFood)
            self.grid.render()
            sleep(0.01) # Animation speed, toggle/change this as desired
        # End simulation if 1 food retrieved while training or if colony died off while not training
        if (self.train and self.colony[0].food > 0) or ((not self.train) and len(self.colony) == 1):
            # Stop!
            print("Ended")
            #self.grid.closeWindow()
            #exit()
            return False # Stop
        return True # Continue


# Test script
# This should be moved to a new file once we get more detailed tests going for training
if __name__ == "__main__":
    print("TEST")
    # Demo world
    world = HexGridWorld(False, 0, animate = True) #, windowSize = (750,500)
    # Run!
    for i in range(3):
        run = True
        while run:
            run = world.step()
        world.reset()
    # Display briefly when done
    #sleep(5)
    print("DONE")
