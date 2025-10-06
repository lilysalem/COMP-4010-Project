from src.hex_grid_helpers import HexPos

class Resource:
    def __init__(self, name: str, path: str, pos: HexPos):
        self.name = name
        self.path = path
        self.pos = pos