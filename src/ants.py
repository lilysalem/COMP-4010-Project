"""
All the ant classes.
The main agents and their behaviours.
Parent ant class plus queen and worker.

"""

# Imports
from hex_grid import HexGrid
from random import randint
from q_learning import QLearningAgent


# Parent ant class
class Ant(object):
    grid: HexGrid # The grid world it's in
    dir: int # direction, 0-5 inclusive, 0 = up, clockwise from there
    x: int # X position
    y: int # Y position
    z: int # Z position
    age: int = 0 # Steps taken, killed by world manager after a while
    
    # Constant array
    # For each direction, the coord of the cell it's facing is its pos plus the corresponding one of these
    visionOffsets = ((1,0,0), (1,1,0), (0,1,0), (0,1,1), (0,0,1), (1,0,1)) # Constant array
    
    # Initialize
    def __init__(self, grid, x: int = 0, y: int = 0, z: int = 0, dir: int = 0):
        # Setup
        self.grid = grid
        self.x = x
        self.y = y
        self.z = z
        self.dir = dir
    
    # Template
    def act(self):
        return

    # Template
    def die(self):
        return


# Queen
class Queen(Ant):
    food: int = 0 # Food received, lets it spawn a worker - 1 to start
    colony: list = None # Access to world manager's list of its colony - first thing in it is always itself

    # Action
    def act(self):
        if self.food > 0: # Attempt to spend a food to spawn a worker in a random adjacent cell if it's empty
            spawn = randint(0, 5)
            cSpawn = self.visionOffsets[spawn]
            if self.grid.getCell(cSpawn) == "E":
                self.dir = spawn
                self.colony.append(Worker(self.grid, x = cSpawn[0], y = cSpawn[1], z = cSpawn[2], dir = (spawn + 3) % 6))
                self.colony[-1].queen = self
                self.grid.setCell(cSpawn, "W")
                self.food -= 1

# Worker
class Worker(Ant):
    hasFood: bool = False
    queen: Queen = None
    q_agent: QLearningAgent = None

    def act(self, action: int = None) -> tuple[tuple[bool, str, str, str], int, int, tuple[bool, str, str, str]]:
        self.age += 1

        vision, visionCoords = self.observe()
        state = (self.hasFood, vision[0], vision[1], vision[2])

        # Q-learning agent must exist for worker to act
        assert self.q_agent is not None, "Worker requires Q-learning agent to act"

        def env_step_func(chosen_action):
            reward = self._execute_action(chosen_action, visionCoords, vision)
            visionNew, _ = self.observe()
            next_state = (self.hasFood, visionNew[0], visionNew[1], visionNew[2])
            return reward, next_state, False, False

        action, reward, stateNew = self.q_agent.step(state, env_step_func)

        return state, action, reward, stateNew

    def _execute_action(self, action: int, visionCoords, vision) -> int:
        if action == 0:
            return self.move(visionCoords[0], vision[0], -1)
        elif action == 1:
            return self.move(visionCoords[1], vision[1], 0)
        elif action == 2:
            return self.move(visionCoords[2], vision[2], 1)
        elif action == 3:
            return self.pickUpFood(visionCoords, vision)
        elif action == 4:
            return self.giveQueenFood(visionCoords)
        else:
            return -1
    
    def observe(self) -> tuple[list[str, str, str], tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]]:
        vOffsets = (self.visionOffsets[self.dir - 1], self.visionOffsets[self.dir], self.visionOffsets[self.dir - 5])
        cSelf = (self.x,self.y,self.z)
        vCoords = (self.grid.add(cSelf, vOffsets[0]), self.grid.add(cSelf, vOffsets[1]), self.grid.add(cSelf, vOffsets[2]))
        vCells = [self.grid.getCell(vCoords[0]), self.grid.getCell(vCoords[1]), self.grid.getCell(vCoords[2])]
        for i in range(3):
            if vCells[i] == "W" or vCells[i] == "V":
                vCells[i] = "O"
            elif vCells[i] == "F" and self.hasFood:
                vCells[i] = "O"
            elif vCells[i] == "Q" and not self.hasFood:
                vCells[i] = "O"
        if self.hasFood:
            cQueen = (self.queen.x, self.queen.y, self.queen.z)
            vDists = [self.grid.distance(vCoords[0], cQueen), self.grid.distance(vCoords[1], cQueen), self.grid.distance(vCoords[2], cQueen)]
            dist = self.grid.distance(cSelf, cQueen)
            for i in range(3):
                if vCells[i] == "E" and vDists[i] < dist:
                    vCells[i] = "S"
        else:
            vTrails = [self.grid.getTrail(vCoords[0]), self.grid.getTrail(vCoords[1]), self.grid.getTrail(vCoords[2])]
            for i in range(3):
                if vCells[i] != "E":
                    vTrails[i] = -1
            maxTrail = max(vTrails)
            for i in range(3):
                if vTrails[i] > 0 and vTrails[i] >= maxTrail / 2:
                    vCells[i] = "T"
        return vCells, vCoords
    
    def move(self, cDest: tuple[int, int, int], cell: str, turn: int) -> int:
        self.dir = (self.dir + turn + 6) % 6
        if self.grid.getCell(cDest) == "E":
            self.grid.setCell(cDest, "W")
            self.grid.setCell((self.x,self.y,self.z), "E")
            if self.hasFood:
                self.grid.addTrail(cDest)
            self.x = cDest[0]
            self.y = cDest[1]
            self.z = cDest[2]
            if cell == "T" or cell == "S":
                return 1
            return 0
        return -1

    def pickUpFood(self, vCoords: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], vCells: list[str, str, str]) -> int:
        if self.hasFood:
            return -1
        if "F" in vCells:
            foodCells = []
            for i in range(3):
                if vCells[i] == "F":
                    foodCells.append(i)
            cFood = vCoords[foodCells[randint(0, len(foodCells) - 1)]]
            self.grid.setCell(cFood, "E")
            self.hasFood = True
            self.grid.addTrail(cFood)
            self.grid.addTrail((self.x,self.y,self.z))
            self.dir = (self.dir + 3) % 6
            return 10
        return -1

    def giveQueenFood(self, vCoords: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> int:
        if not self.hasFood:
            return -1
        cQueen = (self.queen.x, self.queen.y, self.queen.z)
        for i in range(3):
            if vCoords[i] == cQueen:
                self.queen.food += 1
                self.hasFood = False
                return 100
        return -1

    def die(self):
        if self.hasFood:
            self.grid.setCell((self.x, self.y, self.z), "F")
        else:
            self.grid.setCell((self.x, self.y, self.z), "E")

