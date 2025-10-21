"""
Main environment

Includes main for testing (run this one)

Files this env version needs:
-This one
-hexgrid
-ants

"""

import gymnasium as gym
#import numpy as np # Unused while we don't have the reinforcement math, maybe put it in the ant file?
import hex_grid
import ants
import random
from time import sleep


class HexGridWorld(gym.Env):
    xR: int
    yR: int
    zR: int
    stepMax: int
    stepCount: int = 0
    grid: hex_grid.HexGrid
    colony: list = []
    animate: bool = False
    
    def __init__(self, x, y = None, z = None, stepMax = 0, animate = False, windowSize = (1250, 750)):
        self.xR = x
        if y != None:
            self.yR = y
        else:
            self.yR = x
        if z != None:
            self.zR = z
        else:
            self.zR = x
        
        self.stepMax = stepMax

        self.grid = hex_grid.HexGrid(self.xR, self.yR, self.zR)

        if animate:
            self.animate = True
            self.grid.createWindow(windowSize)
        
        print("Started")

        self.generateWorld()

    def generateWorld(self):

        # Random world generation
        maxRockSize = int(min(self.xR, self.yR, self.zR) * 0.15)
        for i in range(self.xR - 1):
            for ii in range(1, self.yR - 1):
                gen = random.randint(0,29)
                if gen == 0:
                    self.grid.setCell((i,ii,0), "F")
                if gen == 1:
                    self.grid.setCell((i,ii,0), "O")
                if gen == 2:
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
        space = 3
        for i in range(space + 1):
            for ii in range(1, space + 1):
                self.grid.setCell((i,ii,0), "E")
                self.grid.setCell((0,i,ii), "E")
                self.grid.setCell((ii,0,i), "E")
        
        # Create queen - ready to run!
        self.colony.append(ants.Queen(self.grid))
        self.colony[0].colony = self.colony
        self.grid.setCell((0,0,0), "Q")

        # Animate
        if self.animate:
            self.grid.drawFullGrid()
            self.grid.render()
            sleep(1)
    
    def buildCluster(self, c, scale, cellType):
        x, y, z = c[0], c[1], c[2]
        if self.grid.isWithinGrid(c):
            self.grid.setCell(c, cellType)
        for i in range(scale):
            for ii in range(1, scale):
                if self.grid.isWithinGrid((x+i,y+ii,z)):
                    self.grid.setCell(self.grid.normalize((x+i,y+ii,z)), cellType)
                if self.grid.isWithinGrid((x,y+i,z+ii)):
                    self.grid.setCell(self.grid.normalize((x,y+i,z+ii)), cellType)
                if self.grid.isWithinGrid((x+ii,y,z+i)):
                    self.grid.setCell(self.grid.normalize((x+ii,y,z+i)), cellType)

    def step(self):
        #print(self.stepCount)

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
            self.grid.drawFullGrid()
            for ant in self.colony:
                if ant == self.colony[0]:
                    self.grid.drawCell((ant.x,ant.y,ant.z), antDir = ant.dir)
                else:
                    self.grid.drawCell((ant.x,ant.y,ant.z), antDir = ant.dir, antHasFood = ant.hasFood)
            self.grid.render()
            sleep(0.01) # Change this as needed
        
        if (self.stepMax >= 1 and self.stepCount >= self.stepMax) or len(self.colony) == 1:
            print("Ended")
            #self.grid.closeWindow()
            #exit()
            return False
        return True


if __name__ == "__main__":
    print("TEST")
    world = HexGridWorld(25, y = 50, z = 75, stepMax = 100000, animate = True) #, windowSize = (750,500)
    run = True
    while run:
        run = world.step()
    sleep(5)
