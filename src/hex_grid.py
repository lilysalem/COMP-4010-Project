"""
Hex grid.
Stores the position of everything.
Contains a lot of getters, setters, and helpers for interacting with the world.
Also handles the Pygame rendering functions, but relies on calls from the world manager.

"""

# Imports
import pygame
from math import sqrt


# Hex grid world
class HexGrid(object):
    xR: int # X range (world dimension, going up)
    yR: int # Y range (world dimension, going right and down)
    zR: int # Z range (world dimension, going left and down)
    cells: list # Special hollowed 3D array of chars, storing object tiles
    trails: list # Special hollowed 3D array of ints, storing trail values

    window: pygame.Surface = None # Display
    windowSize: tuple # Display dimensions
    cellRad: float # For one hexagonal cell, the distance from the centre to an outer point in window pixels (determined from window and grid dimensions, grid must fit in window)
    origin: tuple # Display XY coordinate of grid XYZ coordinate 0,0,0 (where to spawn the grid from given its XYZ dimensions)

    # Initialize
    def __init__(self, xR, yR, zR):
        # World dimensions, passed from world manager
        self.xR = xR
        self.yR = yR
        self.zR = zR
        
        # Create the two hollowed 3D arrays of cells
        # All cells in the hex grid with this system can be represented with at least one dimension of 0 and no negative values
        # Rather than creating a full 3D array, just create the needed parts to save space
        # So instead of a full rectangular prism, just think of it as three adjacent faces of one, like you're viewing it from a corner
        # Start with a 2D XY array, then add a Z array in the first spot in every Y array and every spot in the first Y array
        # Cell map initialized with empty tiles everywhere
        # Trail map initialized to 0 everywhere
        # Cell types:
        # E empty
        # O obstacle
        # F food
        # Q queen
        # W worker
        # V void (out of bounds, only returned from cell getter)
        # Ant adds two more during its state determination
        self.cells = [[["E"] for ii in range(self.yR)] for i in range(self.xR)]
        self.trails = [[[0] for ii in range(self.yR)] for i in range(self.xR)]
        for i in range(self.xR):
            self.cells[i][0] = ["E" for ii in range(self.zR)]
            self.trails[i][0] = [0 for ii in range(self.zR)]
        for i in range(1, self.yR):
            self.cells[0][i] = ["E" for ii in range(self.zR)]
            self.trails[0][i] = [0 for ii in range(self.zR)]

    # Convert and flatten a coordinate to fit within the grid system
    # Catches coords without at least one 0 and with negative values
    # (1,1,1) == (0,0,0), because they cancel, and (-1,0,0) == (0,1,1) for all axes
    # So a coord can be adjusted the same on every value
    def normalize(self, c):
        m = min(c)
        return (c[0]-m,c[1]-m,c[2]-m)
    
    # Checker
    def isWithinGrid(self, c):
        cN = self.normalize(c)
        return cN[0] < self.xR and cN[1] < self.yR and cN[2] < self.zR
    
    # Coord addition
    def add(self, c1, c2):
        return self.normalize((c1[0]+c2[0],c1[1]+c2[1],c1[2]+c2[2]))

    # Distance of shortest cell path between coords, ignoring tiles
    def distance(self, c1, c2):
        diff = (c2[0]-c1[0],c2[1]-c1[1],c2[2]-c1[2])
        return max(self.normalize(diff))
    
    # Getter
    def getCell(self, c):
        if c[0] >= self.xR or c[1] >= self.yR or c[2] >= self.zR:
            return "V" # Void, off the grid
        return self.cells[c[0]][c[1]][c[2]]
    
    # Getter
    def getTrail(self, c):
        if c[0] >= self.xR or c[1] >= self.yR or c[2] >= self.zR:
            return 0
        return self.trails[c[0]][c[1]][c[2]]
    
    # Setter
    def setCell(self, c, new):
        self.cells[c[0]][c[1]][c[2]] = new
    
    # Setter, currently unused
    def setTrail(self, c, new):
        self.trails[c[0]][c[1]][c[2]] = new
    
    # Setter with default value for new trails
    def addTrail(self, c):
        self.trails[c[0]][c[1]][c[2]] = 250 # This is changeable
    
    # Reduce the strength of the trail at a cell by 1
    def fadeTrail(self, c):
        if self.trails[c[0]][c[1]][c[2]] > 0:
            self.trails[c[0]][c[1]][c[2]] -= 1
    
    # Fade trail over whole grid, called every step
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
    
    # Initialize Pygame window, called if simulation is animated - doesn't affect internals
    def createWindow(self, windowSize):
        # Setup
        self.windowSize = windowSize
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(windowSize)
        
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
    
    # Render the window
    def render(self):
        pygame.display.update()
        return
    
    # Update the display for one cell
    def drawCell(self, c, antDir = 0, antHasFood = False):
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
        cell = self.getCell(c)
        trail = self.getTrail(c)

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
    
    # Convert an XYZ grid coord to an XY window coord
    # I hate math
    def convertGridCoord(self, c):
        wX = self.origin[0] + ((c[1] - c[2]) * 1.5 * self.cellRad)
        wY = self.origin[1] - (((c[0] * 2) - c[1] - c[2]) * sqrt(0.75) * self.cellRad)
        return (wX,wY)

# Old testing
if __name__ == "__main__":
    testGrid = HexGrid(3, 3, 3)
    print(testGrid.cells)
