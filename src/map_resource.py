from dataclasses import dataclass
from src.hex_grid_helpers import HexPos

@dataclass
class MapResource:
    name: str
    pos: HexPos
    type: str
