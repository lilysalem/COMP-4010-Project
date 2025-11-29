"""
Hex grid.
Stores the position of everything.
Contains a lot of getters, setters, and helpers for interacting with the world.
Also handles the Pygame rendering functions, but relies on calls from the world manager.

"""

# Imports


# Hex grid world
class HexGrid(object):
    xR: int # X range (world dimension, going up)
    yR: int # Y range (world dimension, going right and down)
    zR: int # Z range (world dimension, going left and down)
    cells: list # Special hollowed 3D array of chars, storing object tiles
    trails: list # Special hollowed 3D array of ints, storing trail values

    # Initialize
    def __init__(self, xR: int, yR: int, zR: int):
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
    # A coord can be adjusted with the same formula for all inputs which is nice
    def normalize(self, c: tuple[int, int, int]) -> tuple[int, int, int]:
        m = min(c)
        return (c[0]-m,c[1]-m,c[2]-m)
    
    # Checker
    def isWithinGrid(self, c: tuple[int, int, int]) -> bool:
        cN = self.normalize(c)
        return cN[0] < self.xR and cN[1] < self.yR and cN[2] < self.zR
    
    # Coord addition
    def add(self, c1: tuple[int, int, int], c2: tuple[int, int, int]) -> tuple[int, int, int]:
        return self.normalize((c1[0]+c2[0],c1[1]+c2[1],c1[2]+c2[2]))

    # Distance of shortest cell path between coords, ignoring tile types
    def distance(self, c1: tuple[int, int, int], c2: tuple[int, int, int]) -> int:
        diff = (c2[0]-c1[0],c2[1]-c1[1],c2[2]-c1[2])
        return max(self.normalize(diff))
    
    # Getter
    def getCell(self, c: tuple[int, int, int]) -> str:
        if c[0] >= self.xR or c[1] >= self.yR or c[2] >= self.zR:
            return "V" # Void, off the grid
        return self.cells[c[0]][c[1]][c[2]]
    
    # Getter
    def getTrail(self, c: tuple[int, int, int]) -> int:
        if c[0] >= self.xR or c[1] >= self.yR or c[2] >= self.zR:
            return 0
        return self.trails[c[0]][c[1]][c[2]]
    
    # Setter
    def setCell(self, c: tuple[int, int, int], new: str):
        self.cells[c[0]][c[1]][c[2]] = new
    
    # Setter
    def setTrail(self, c: tuple[int, int, int], new: str):
        self.trails[c[0]][c[1]][c[2]] = new
    
    # Setter with default value for new trails
    def addTrail(self, c: tuple[int, int, int]):
        self.trails[c[0]][c[1]][c[2]] = 250 # This is changeable
    
    # Reduce the strength of the trail at a cell by 1
    def fadeTrail(self, c: tuple[int, int, int]):
        if self.trails[c[0]][c[1]][c[2]] > 0:
            self.trails[c[0]][c[1]][c[2]] -= 1
    
    # Fade trail over whole grid
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
