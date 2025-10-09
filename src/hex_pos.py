from dataclasses import dataclass


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
            raise TypeError(f"Cannot compare HexPos with {type(other)}")
        return (self.q, self.r, self.s) == (other.q, other.r, other.s)
    
    def distance(self, other):
        return ((abs(self.q - other.q) + abs(self.r - other.r) + abs(self.s - other.s)) // 2)