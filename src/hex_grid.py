"""
Hex grid

Also handles the Pygame rendering functions, but relies on calls from the environment

"""

import pygame
from math import sqrt


class HexGrid(object):
    xR: int
    yR: int
    zR: int
    cells: list
    trails: list

    window: pygame.Surface = None
    windowSize: tuple
    cellRad: float
    origin: tuple

    def __init__(self, xR, yR, zR):
        self.xR = xR
        self.yR = yR
        self.zR = zR
        
        self.cells = [[["E"] for ii in range(self.yR)] for i in range(self.xR)]
        for i in range(self.xR):
            self.cells[i][0] = ["E" for ii in range(self.zR)]
        for i in range(1, self.yR):
            self.cells[0][i] = ["E" for ii in range(self.zR)]
        
        self.trails = [[[0] for ii in range(self.yR)] for i in range(self.xR)]
        for i in range(self.xR):
            self.trails[i][0] = [0 for ii in range(self.zR)]
        for i in range(1, self.yR):
            self.trails[0][i] = [0 for ii in range(self.zR)]

    def normalize(self, c):
        m = min(c)
        return (c[0]-m,c[1]-m,c[2]-m)
    
    def isWithinGrid(self, c):
        cN = self.normalize(c)
        return cN[0] < self.xR and cN[1] < self.yR and cN[2] < self.zR
    
    def add(self, c1, c2):
        return self.normalize((c1[0]+c2[0],c1[1]+c2[1],c1[2]+c2[2]))

    def distance(self, c1, c2):
        diff = (c2[0]-c1[0],c2[1]-c1[1],c2[2]-c1[2])
        return max(self.normalize(diff))
    
    def getCell(self, c):
        if c[0] >= self.xR or c[1] >= self.yR or c[2] >= self.zR:
            return "V"
        return self.cells[c[0]][c[1]][c[2]]
        
    def getTrail(self, c):
        if c[0] >= self.xR or c[1] >= self.yR or c[2] >= self.zR:
            return 0
        return self.trails[c[0]][c[1]][c[2]]
    
    def setCell(self, c, new):
        self.cells[c[0]][c[1]][c[2]] = new
    
    def setTrail(self, c, new):
        self.trails[c[0]][c[1]][c[2]] = new
    
    def addTrail(self, c):
        self.trails[c[0]][c[1]][c[2]] = 250
    
    def fadeTrail(self, c):
        if self.trails[c[0]][c[1]][c[2]] > 0:
            self.trails[c[0]][c[1]][c[2]] -= 1
    
    def fadeAllTrails(self):
        self.fadeTrail((0,0,0))
        for i in range(self.xR):
            for ii in range(1, self.yR):
                self.fadeTrail((i,ii,0))
        for i in range(self.yR):
            for ii in range(1, self.zR):
                self.fadeTrail((0,i,ii))
        for i in range(self.zR):
            for ii in range(1, self.xR):
                self.fadeTrail((ii,0,i))
    
    def createWindow(self, windowSize):
        self.windowSize = windowSize
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(windowSize)
        
        widthInCellRads = (self.yR + self.zR) * 1.5 - 1
        heightInCellRads = ((self.xR * 2) + self.yR + self.zR - 2) * sqrt(0.75)
        widthMaxCellRad = (windowSize[0] - 10) / widthInCellRads
        heightMaxCellRad = (windowSize[1] - 10) / heightInCellRads
        self.cellRad = min(widthMaxCellRad, heightMaxCellRad)

        originXOffsetInCellRads = (widthInCellRads / 2) - (self.yR * 1.5 - 0.5)
        originYOffsetInCellRads = (heightInCellRads / 2) - ((self.yR + self.zR - 1) * sqrt(0.75))
        originX = (windowSize[0] / 2) + (originXOffsetInCellRads * self.cellRad)
        originY = (windowSize[1] / 2) + (originYOffsetInCellRads * self.cellRad)
        self.origin = (originX,originY)
    
    def closeWindow(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def render(self):
        pygame.display.update()
        return
    
    def drawFullGrid(self):
        self.drawCell((0,0,0))
        for i in range(self.xR):
            for ii in range(1, self.yR):
                self.drawCell((i,ii,0))
        for i in range(self.yR):
            for ii in range(1, self.zR):
                self.drawCell((0,i,ii))
        for i in range(self.zR):
            for ii in range(1, self.xR):
                self.drawCell((ii,0,i))
    
    def drawCell(self, c, antDir = 0, antHasFood = False):
        wXY = self.convertGridCoord(c)
        wX, wY = wXY[0], wXY[1]
        r = self.cellRad
        rH = r / 2
        rS = r * sqrt(0.75)
        p1 = (wX - r, wY)
        p2 = (wX - rH, wY - rS)
        p3 = (wX + rH, wY - rS)
        p4 = (wX + r, wY)
        p5 = (wX + rH, wY + rS)
        p6 = (wX - rH, wY + rS)
        hexagon = (p1, p2, p3, p4, p5, p6)

        cell = self.getCell(c)
        trail = self.getTrail(c)

        hexagonColour = (63,63,63)
        if cell != "O":
            t = max(0, 255 - trail)
            hexagonColour = (t,255,t)
        pygame.draw.polygon(self.window, hexagonColour, hexagon)

        if cell == "Q":
            pygame.draw.circle(self.window, (255,0,0), wXY, rS)
        elif cell == "W":
            pygame.draw.circle(self.window, (191,0,0), wXY, rS)
            if antHasFood:
                pygame.draw.circle(self.window, (255,127,0), wXY, rH)
        elif cell == "F":
            pygame.draw.circle(self.window, (255,127,0), wXY, rH)
    
    def convertGridCoord(self, c):
        wX = self.origin[0] + ((c[1] - c[2]) * 1.5 * self.cellRad)
        wY = self.origin[1] - (((c[0] * 2) - c[1] - c[2]) * sqrt(0.75) * self.cellRad)
        return (wX,wY)


if __name__ == "__main__":
    # testing
    testGrid = HexGrid(3, 3, 3)
    print(testGrid.cells)
