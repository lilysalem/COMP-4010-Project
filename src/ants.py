"""
All the ant classes.
The main agents and their behaviours.
Parent ant class plus queen and worker.

"""

# Imports
from random import randint
from hex_grid import HexGrid


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
    def __init__(self, grid, dir = 0, x = 0, y = 0, z = 0):
        # Setup
        self.grid = grid
        self.dir = dir
        self.x = x
        self.y = y
        self.z = z
    
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
                #self.grid.drawCell(cSpawn)
                self.food -= 1

# Worker
class Worker(Ant):
    hasFood: bool = False # False: foraging, true: retrieving
    queen: Queen = None # Queen of its colony

    # Action
    def act(self, action = None):
        # Increment age
        self.age += 1

        # State
        vision, visionCoords = self.observe()
        state = (self.hasFood, vision[0], vision[1], vision[2])
        # State to be analyzed is the three cells in vision plus hasFood, so: char, char, char, bool
        # Will probably need to be converted to a different data type for math analysis
        
        # Toggle: random action or my preset algorithm mimicking good performance.
        # This will have to be reworked once we get the actual reinforcement learning going.
        if action == None:
            """ # (Comment/uncomment this line to toggle)
            action = random.randint(1, 5)
            """
            if "Q" in vision and self.hasFood:
                action = 4
            elif "F" in vision and not self.hasFood:
                action = 3
            elif self.hasFood and ("S" not in vision) and (vision[1] == "O"):
                if vision[0] == vision[2]:
                    action = randint(0,1) * 2
                elif vision[0] == "O":
                    action = 2
                elif vision[2] == "O":
                    action = 0
            else:
                bestPathActions = []
                for i in range(3):
                    if vision[i] == "T" or vision[i] == "S":
                        bestPathActions.append(i)
                if len(bestPathActions) > 0:
                    action = bestPathActions[randint(0, len(bestPathActions) - 1)]
                else:
                    action = randint(0, 2)
            #"""

        # Take the action and get the reward
        if action == 0:
            reward = self.move(visionCoords[0], vision[0], -1)
        if action == 1:
            reward = self.move(visionCoords[1], vision[1], 0)
        if action == 2:
            reward = self.move(visionCoords[2], vision[2], 1)
        if action == 3:
            reward = self.pickUpFood(visionCoords, vision)
        if action == 4:
            reward = self.giveQueenFood(visionCoords)

        # New state
        visionNew, visionCoordsNew = self.observe()
        stateNew = (self.hasFood, visionNew[0], visionNew[1], visionNew[2])

        return state, action, reward, stateNew
    
    # Observe cells in vision range
    def observe(self):
        # Get and process the cells in its vision range
        # It sees three adjacent cells, detemined by its direction
        # Two more cell types which could replace an E:
        # T trail (E cells with strongest trail values relative to each other)
        # S shortest path (E cell with shorter distance to queen than current cell)
        # T is only used when foraging
        # S is only used when returning
        # Result: three chars representing the post-process cells in its vision
        vOffsets = (self.visionOffsets[self.dir - 1], self.visionOffsets[self.dir], self.visionOffsets[self.dir - 5])
        cSelf = (self.x,self.y,self.z)
        vCoords = (self.grid.add(cSelf, vOffsets[0]), self.grid.add(cSelf, vOffsets[1]), self.grid.add(cSelf, vOffsets[2]))
        vCells = [self.grid.getCell(vCoords[0]), self.grid.getCell(vCoords[1]), self.grid.getCell(vCoords[2])]
        for i in range(3):
            # Cut down the state space by eliminating redundant ones
            # Treat cells occupied by other workers and cells outside the world bounds as obstacles
            # If already has food, do the same for food objects because nothing happens on interaction with it in that mode
            # If looking for food, do the same for the queen because nothing happens on interaction with it in that mode
            # This isn't really necessary, it's just a shortcut, but it'll make training faster
            if vCells[i] == "W" or vCells[i] == "V":
                vCells[i] = "O"
            elif vCells[i] == "F" and self.hasFood:
                vCells[i] = "O"
            elif vCells[i] == "Q" and not self.hasFood:
                vCells[i] = "O"
        if self.hasFood:
            # Any E cell which would make the ant closer to the queen if moved into is converted to S
            cQueen = (self.queen.x, self.queen.y, self.queen.z)
            vDists = [self.grid.distance(vCoords[0], cQueen), self.grid.distance(vCoords[1], cQueen), self.grid.distance(vCoords[2], cQueen)]
            #minDist = min(vDists)
            dist = self.grid.distance(cSelf, cQueen)
            for i in range(3):
                #if vCells[i] == "E" and vDists[i] == minDist:
                if vCells[i] == "E" and vDists[i] < dist:
                    vCells[i] = "S"
        else:
            # The E cell with the highest trail value and other E cells with a value at least half that value are converted to T
            vTrails = [self.grid.getTrail(vCoords[0]), self.grid.getTrail(vCoords[1]), self.grid.getTrail(vCoords[2])]
            for i in range(3):
                if vCells[i] != "E":
                    vTrails[i] = -1
            maxTrail = max(vTrails)
            for i in range(3):
                #if vTrails[i] == maxTrail:
                #if vTrails[i] > 0:
                if vTrails[i] > 0 and vTrails[i] >= maxTrail / 2:
                    vCells[i] = "T"
        return vCells, vCoords
    
    # Move to a cell
    def move(self, cDest, cell, turn):
        self.dir = (self.dir + turn + 6) % 6
        if self.grid.getCell(cDest) == "E": # Check cell on the actual grid to make sure it's empty
            self.grid.setCell(cDest, "W")
            self.grid.setCell((self.x,self.y,self.z), "E")
            if self.hasFood:
                self.grid.addTrail(cDest)
            #self.grid.drawCell((self.x,self.y,self.z))
            #self.grid.drawCell(cDest, antHasFood = self.hasFood)
            self.x = cDest[0]
            self.y = cDest[1]
            self.z = cDest[2]
            if cell == "T" or cell == "S":
                return 1 # Good reward for following a trail or seeking queen
            return 0 # Neutral reward for moving otherwise
        return -1 # Bad reward for trying to move into an occupied cell, wasted action
    
    # Pick up food
    # Succeeds if a food object is in any vision cell
    # If multiple are there, choose one at random
    def pickUpFood(self, vCoords, vCells):
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
            #self.grid.drawCell((self.x,self.y,self.z), antHasFood = True)
            #self.grid.drawCell(cFood)
            return 10 # Big reward for picking up food
        return -1 # Bad reward if no food in range, wasted action
    
    # Give food to the queen
    # Succeeds if the queen is in any vision cell
    def giveQueenFood(self, vCoords):
        if not self.hasFood:
            return -1
        cQueen = (self.queen.x, self.queen.y, self.queen.z)
        for i in range(3):
            if vCoords[i] == cQueen:
                self.queen.food += 1
                self.hasFood = False
                return 100 # Huge reward for giving food
        return -1 # Bad reward if queen not in range, wasted action
    
    # Die when killed by world manager when lifespan hits the cap
    def die(self):
        if self.hasFood:
            self.grid.setCell((self.x, self.y, self.z), "F")
        else:
            self.grid.setCell((self.x, self.y, self.z), "E")

# Soldier
class Soldier(Ant):
    # Unused for now, maybe later - probably not though
    def act(self):
        print("KILL!")
