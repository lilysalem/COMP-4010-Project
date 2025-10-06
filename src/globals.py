from enum import Enum
from src.hex_grid_helpers import HexPos

class TERRAIN_TYPES(Enum):
    grass = 1
    water = 2
    wall = 3
    void = 4

class RESOURCE_TYPES(Enum):
    food = 1

DIRECTIONS = [
        HexPos(1, -1, 0), HexPos(1, 0, -1), HexPos(0, 1, -1),
        HexPos(-1, 1, 0), HexPos(-1, 0, 1), HexPos(0, -1, 1)
    ]