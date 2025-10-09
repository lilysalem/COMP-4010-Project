from dataclasses import dataclass
from src.data_structures import HexPos

@dataclass
class MapResource:
    name: str
    pos: HexPos
    type: str
