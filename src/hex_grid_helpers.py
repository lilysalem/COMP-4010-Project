import numpy as np
import src.globals as GLOBALS
from dataclasses import dataclass
from src.hex_pos import HexPos


@dataclass
class HexCell:
    pos: HexPos
    terrain: GLOBALS.TERRAIN_TYPES = GLOBALS.TERRAIN_TYPES.grass  # Using default value instead of imported constant
    cost: int = 1

    def distance(self, pos: HexPos):
        return self.pos.distance(pos)

    def blocked(self):
        return self.terrain == GLOBALS.TERRAIN_TYPES.void or self.terrain == GLOBALS.TERRAIN_TYPES.wall


@dataclass
class MapResource:
    name: str
    pos: HexPos
    type: str


@dataclass
class HexMap:
    q_limit: int
    r_limit: int
    s_limit: int
    seed: str = ""
    spawn_point: HexPos = HexPos(0, 0, 0)
    target_point = None  # Will be defined in __post_init__
    resources: list[MapResource] = None
    cells: dict[HexPos, HexCell] = None

    def __post_init__(self):
        if self.resources is None:
            self.resources = []
        if self.cells is None:
            self.cells = {}
        
        for q in range(-self.q_limit, self.q_limit + 1):
            for r in range(-self.r_limit, self.r_limit + 1):
                s = -q - r
                if abs(s) <= self.s_limit:
                    pos = HexPos(q, r, s)
                    # Randomly assign terrain types based on some probabilities
                    
                    self.cells[pos] = HexCell(pos=pos, terrain=GLOBALS.TERRAIN_TYPES.grass)
        
        # Convert string seed to integer hash for numpy
        if self.seed:
            np.random.seed(abs(hash(self.seed)) % (2**32))
        else:
            np.random.seed(None)  # Use system time if no seed provided
        # check validity for provided manual resources
        for resource in self.resources:
            pos = resource.pos
            if (abs(pos.q) > self.q_limit or
                abs(pos.r) > self.r_limit or
                abs(pos.s) > self.s_limit):
                raise ValueError(f"Resource {resource.name} at {pos} is out of map bounds.")
            cell = self.cells.get(pos)
            if cell and cell.terrain == GLOBALS.TERRAIN_TYPES.void:
                raise ValueError(f"Resource {resource.name} at {pos} is placed on void terrain.")

        q = np.random.randint(-self.q_limit, self.q_limit + 1)
        r = np.random.randint(-self.r_limit, self.r_limit + 1)
        s = -q - r  # Ensure q + r + s = 0
        self.target_point = HexPos(q, r, s)
        print("post init")
    
    def is_pos_valid(self, hex_pos: HexPos):
        pos_cell = self.cells.get(hex_pos)
        return (abs(hex_pos.q) <= self.q_limit and
                abs(hex_pos.r) <= self.r_limit and
                abs(hex_pos.s) <= self.s_limit and
                pos_cell is not None and
                not pos_cell.blocked)

    def neighbors(self, hex_pos: HexPos):
        neighbors = []
        for direction in GLOBALS.DIRECTIONS:
            neighbor = HexPos(hex_pos.q + direction.q,
                              hex_pos.r + direction.r,
                              hex_pos.s + direction.s)
            if (abs(neighbor.q) <= self.q_limit and
                abs(neighbor.r) <= self.r_limit and
                abs(neighbor.s) <= self.s_limit):
                neighbors.append(neighbor)
        return neighbors