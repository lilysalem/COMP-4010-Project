from enum import Enum
from src.hex_pos import HexPos
from dataclasses import dataclass


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

COLORS = {
    TERRAIN_TYPES.grass: (34, 139, 34),   # ForestGreen
    TERRAIN_TYPES.water: (30, 144, 255),  # Dodger  Blue
    TERRAIN_TYPES.wall: (105, 105, 105),   # DimGray
    TERRAIN_TYPES.void: (0, 0, 0),         # Black
}