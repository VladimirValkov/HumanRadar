import math


class trackedObject:
    def __init__(self, X, Y, ID):
        self.id = ID
        self.x = X
        self.y = Y
        self.frames = 0
        self.zone1 = 0
        self.zone2 = 0

    def insideRadius(self, newX, newY, r=20.0) -> bool:
        formula = math.sqrt(((self.x - newX) * (self.x - newX)) + ((self.y - newY) * (self.y - newY)))
        if formula <= r:
            return True
        else:
            return False

    def update(self, newX, newY):
        self.x = newX
        self.y = newY

    def count(self):
        self.frames += 1

    def clear(self):
        self.frames = 0

    def zone(self, time):
        if self.y <= 800 / 3:
            self.zone1 = time
        elif self.y >= 800 / 2:
            self.zone2 = time