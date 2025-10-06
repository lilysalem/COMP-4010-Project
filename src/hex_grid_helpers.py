from dataclasses import dataclass
from src.resource import Resource
import src.globals as GLOBALS


@dataclass(frozen=True, order=True)
class HexPos:
    q: int  # Column
    r: int  # Row
    s: int  # fuck off copilot stop adding comments if you don't know what's goin on

    def __post_init__(self):
        if self.q + self.r + self.s != 0:
            raise ValueError(f"Invalid cube coord {self} (q+r+s must be 0).")

    def __eq__(self, other):
        if not isinstance(other, HexPos):
            return NotImplemented
        return (self.q, self.r, self.s) == (other.q, other.r, other.s)

@dataclass
class HexCell:
    pos: HexPos
    terrain: GLOBALS.TERRAIN_TYPES.grass
    cost: int = 1
    
    def blocked(self):
        return self.terrain == GLOBALS.TERRAIN_TYPES.void or self.terrain == GLOBALS.TERRAIN_TYPES.wall

@dataclass
class HexMap:
    q_limit: int
    r_limit: int
    s_limit: int
    seed: str = ""
    resources: list[Resource] = []
    cells: dict[HexPos, HexCell] = {}

    def __post_init__(self):
        # check validity for provided manual resources
        for resource in self.resources:
            pos = resource.pos
            if (abs(pos.q) > self.q_limit or
                abs(pos.r) > self.r_limit or
                abs(pos.s) > self.s_limit):
                raise ValueError(f"Resource {resource.name} at {pos} is out of map bounds.")
            cell = self.cells.get(pos)
            if cell.terrain == GLOBALS.TERRAIN_TYPES.void:
                raise ValueError(f"Resource {resource.name} at {pos} is placed on void terrain.")
        # fill in resource and terrain generation here
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