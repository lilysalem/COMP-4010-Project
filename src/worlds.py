"""
World generation functions.
Each world preset function is one layout.
Also includes some helpers.

"""

# Imports
from hex_grid import HexGrid
from hex_grid_world import HexGridWorld
import ants
from random import randint


# Helper for placing a queen
def createQueen(world: HexGridWorld, c: tuple[int, int, int]):
    world.colony.append(ants.Queen(world.grid, x = c[0], y = c[1], z = c[2]))
    world.colony[0].colony = world.colony
    world.grid.setCell(c, "Q")
    
# Helper for placing a worker
def createWorker(world: HexGridWorld, c: tuple[int, int, int], dir: int = 0):
    world.colony.append(ants.Worker(world.grid, x = c[0], y = c[1], z = c[2], dir = dir))
    world.colony[-1].queen = world.colony[0]
    world.grid.setCell(c, "W")

# Helper for filling hexagonal clusters of cells with a tile type
def buildCluster(world: HexGridWorld, c: tuple[int, int, int], scale: int, cellType: str):
    x, y, z = c[0], c[1], c[2]
    # Centre cell
    if world.grid.isWithinGrid(c):
        world.grid.setCell(c, cellType)
    # Everything else
    for i in range(scale):
        for ii in range(1, scale):
            if world.grid.isWithinGrid((x+i,y+ii,z)):
                world.grid.setCell(world.grid.normalize((x+i,y+ii,z)), cellType)
            if world.grid.isWithinGrid((x,y+i,z+ii)):
                world.grid.setCell(world.grid.normalize((x,y+i,z+ii)), cellType)
            if world.grid.isWithinGrid((x+ii,y,z+i)):
                world.grid.setCell(world.grid.normalize((x+ii,y,z+i)), cellType)


# World 0 (default random)
# Random dimensions, terrain, food
# Stores the world on the first episode and loads it on subsequent
def randomWorld(world: HexGridWorld):
    # First episode
    if world.gridMemory == None:
        # New random map, first episode
        if world.xR == None:
            world.xR = randint(10, 100)
        if world.yR == None:
            world.yR = randint(10, 100)
        if world.zR == None:
            world.zR = randint(10, 100)
        # Create grid
        world.grid = HexGrid(world.xR, world.yR, world.zR)
        # Random world generation
        maxRockSize = int(min(world.xR, world.yR, world.zR) * 0.1)
        # Three nested loops, each covering the sector with the two used axes plus one of those axes
        for i in range(world.xR):
            for ii in range(1, world.yR):
                gen = randint(0,29) # Randomly determine what object to place
                if gen == 0: # Food
                    world.grid.setCell((i,ii,0), "F")
                if gen == 1: # Obstacle
                    world.grid.setCell((i,ii,0), "O")
                if gen == 2: # Obstacle but larger cluster
                    buildCluster(world, (i,ii,0), randint(1, maxRockSize), "O")
        for i in range(world.yR):
            for ii in range(1, world.zR):
                gen = randint(0,29)
                if gen == 0:
                    world.grid.setCell((0,i,ii), "F")
                if gen == 1:
                    world.grid.setCell((0,i,ii), "O")
                if gen == 2:
                    buildCluster(world, (0,i,ii), randint(1, maxRockSize), "O")
        for i in range(world.zR):
            for ii in range(1, world.xR):
                gen = randint(0,29)
                if gen == 0:
                    world.grid.setCell((ii,0,i), "F")
                if gen == 1:
                    world.grid.setCell((ii,0,i), "O")
                if gen == 2:
                    buildCluster(world, (ii,0,i), randint(1, maxRockSize), "O")
        # Large food clusters
        # Three, one close to the end of each axis, size also dependent on that axis
        pileX = int(world.xR * 0.75)
        sizeX = int((world.xR - pileX) * 0.5)
        pileY = int(world.yR * 0.75)
        sizeY = int((world.yR - pileY) * 0.5)
        pileZ = int(world.zR * 0.75)
        sizeZ = int((world.zR - pileZ) * 0.5)
        buildCluster(world, (pileX,0,0), sizeX, "F")
        buildCluster(world, (0,pileY,0), sizeY, "F")
        buildCluster(world, (0,0,pileZ), sizeZ, "F")
        # Clear space for colony
        # If they don't have any they get into a traffic jam
        space = 3 # Number of free cells around the queen on each axis
        buildCluster(world, (0,0,0), space + 1, "E")
        # Store map
        world.gridMemory = ["Q"]
        for i in range(world.xR):
            for ii in range(1, world.yR):
                world.gridMemory.append(world.grid.getCell((i,ii,0)))
        for i in range(world.yR):
            for ii in range(1, world.zR):
                world.gridMemory.append(world.grid.getCell((0,i,ii)))
        for i in range(world.zR):
            for ii in range(1, world.xR):
                world.gridMemory.append(world.grid.getCell((ii,0,i)))
    # Subsequent episodes
    else:
        # Load previously saved map
        world.grid = HexGrid(world.xR, world.yR, world.zR)
        world.grid.setCell((0,0,0), world.gridMemory[0])
        iterateCell = 1
        for i in range(world.xR):
            for ii in range(1, world.yR):
                world.grid.setCell((i,ii,0), world.gridMemory[iterateCell])
                iterateCell += 1
        for i in range(world.yR):
            for ii in range(1, world.zR):
                world.grid.setCell((0,i,ii), world.gridMemory[iterateCell])
                iterateCell += 1
        for i in range(world.zR):
            for ii in range(1, world.xR):
                world.grid.setCell((ii,0,i), world.gridMemory[iterateCell])
                iterateCell += 1
    # Create queen
    # Always at 0,0,0 in random world
    createQueen(world, (0,0,0))
    world.colony[0].food = 1 # One worker to start
    world.colony[0].act() # Spawn that worker at random adjacent location

# World 1
# Small, queen at bottom, one food at top, pre-drawn trail straight between them
def presetWorld1(world: HexGridWorld):
    world.xR = 10
    world.yR = 10
    world.zR = 10
    world.grid = HexGrid(world.xR, world.yR, world.zR)
    world.grid.setCell((9,0,0), "F")
    for i in range(-8, 9):
        world.grid.setTrail(world.grid.normalize((i,0,0)), 25)
    createQueen(world, (0,9,9))
    createWorker(world, (0,8,8))
