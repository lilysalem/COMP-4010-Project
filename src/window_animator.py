"""
Pygame animation window handler.

"""

# Imports
from hex_grid import HexGrid
import pygame
from math import sqrt
import time


# Animator with frame rate control
class Animator(object):
    TARGET_FPS = 30  # set framerate here - it doesn't affect the training speed, but does take hella resources

    xR: int
    yR: int
    zR: int
    window: pygame.Surface = None
    windowSize: tuple
    cellRad: float
    origin: tuple
    last_frame_time: float = 0
    
    # Initialize
    def __init__(self, xR: int, yR: int, zR: int, windowSize: tuple[int, int]):
        self.xR = xR
        self.yR = yR
        self.zR = zR
        self.last_frame_time = time.time()
        self.createWindow(windowSize)
    
    # Initialize Pygame window, called if simulation is animated - doesn't affect internals
    def createWindow(self, windowSize: tuple[int, int]):
        # Setup
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(windowSize)
        self.windowSize = windowSize
        
        # Calculate how big to make each cell so it fits the window right
        # I hate math
        widthInCellRads = (self.yR + self.zR) * 1.5 - 1
        heightInCellRads = ((self.xR * 2) + self.yR + self.zR - 2) * sqrt(0.75)
        widthMaxCellRad = (windowSize[0] - 10) / widthInCellRads
        heightMaxCellRad = (windowSize[1] - 10) / heightInCellRads
        self.cellRad = min(widthMaxCellRad, heightMaxCellRad)

        # Calulate grid positioning in the window
        # I hate math
        originXOffsetInCellRads = (widthInCellRads / 2) - (self.yR * 1.5 - 0.5)
        originYOffsetInCellRads = (heightInCellRads / 2) - ((self.yR + self.zR - 1) * sqrt(0.75))
        originX = (windowSize[0] / 2) + (originXOffsetInCellRads * self.cellRad)
        originY = (windowSize[1] / 2) + (originYOffsetInCellRads * self.cellRad)
        self.origin = (originX,originY)
    
    # Close window, currently unused - gotta fix this
    def closeWindow(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def updateWindow(self):
        current_time = time.time()
        frame_time = 1.0 / self.TARGET_FPS
        elapsed = current_time - self.last_frame_time

        if elapsed >= frame_time:
            pygame.display.update()
            self.last_frame_time = current_time
    
    # Update the display for one cell
    def drawCell(self, grid: HexGrid, c: tuple[int, int, int], antDir: int = 0, antHasFood: bool = False):
        # Where to put it
        wXY = self.convertGridCoord(c)
        wX, wY = wXY[0], wXY[1]

        # Points for the hexagon polygon
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

        # Convenience
        cell = grid.getCell(c)
        trail = grid.getTrail(c)

        # Cell background
        hexagonColour = (63,63,63) # Obstacle, default
        if cell != "O": # Show trail
            t = max(0, 255 - trail)
            hexagonColour = (t,255,t)
        pygame.draw.polygon(self.window, hexagonColour, hexagon)

        # On top of background
        if cell == "Q": # Queen
            pygame.draw.circle(self.window, (255,0,0), wXY, rS)
        elif cell == "W": # Worker, can be shown carrying food
            pygame.draw.circle(self.window, (191,0,0), wXY, rS)
            if antHasFood:
                pygame.draw.circle(self.window, (255,127,0), wXY, rH)
        elif cell == "F": # Food
            pygame.draw.circle(self.window, (255,127,0), wXY, rH)
    
    # Update the whole world display
    def drawFullGrid(self, grid: HexGrid):
        self.drawCell(grid, (0,0,0))
        for i in range(self.xR):
            for ii in range(1, self.yR):
                self.drawCell(grid, (i,ii,0))
        for i in range(self.yR):
            for ii in range(1, self.zR):
                self.drawCell(grid, (0,i,ii))
        for i in range(self.zR):
            for ii in range(1, self.xR):
                self.drawCell(grid, (ii,0,i))
    
    # Convert an XYZ grid coord to an XY window coord
    # I hate math
    def convertGridCoord(self, c: tuple[int, int, int]) -> tuple[int, int]:
        wX = self.origin[0] + ((c[1] - c[2]) * 1.5 * self.cellRad)
        wY = self.origin[1] - (((c[0] * 2) - c[1] - c[2]) * sqrt(0.75) * self.cellRad)
        return (wX,wY)
