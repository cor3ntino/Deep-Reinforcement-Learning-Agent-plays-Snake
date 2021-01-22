import numpy as np
import random
from constants import *

class Snake:
    
    def __init__(self):
        self.body = [[3, random.randint(GAME_DIMENSION // 2 - 1, GAME_DIMENSION // 2 + 1)]]
        self.body.append([self.body[0][0] - 1, self.body[0][1]])
        self.body.append([self.body[0][0] - 2, self.body[0][1]])
        self.old_tail = self.body[-1]
        self.direction = [1, 0]
        self.age = 0
        self.starve = 100
        self.alive = True

    def move(self):
        head = self.body[0]
        self.old_tail = self.body.pop()
        self.body.insert(0, [])
        self.body[0] = [head[0] + self.direction[0], head[1] + self.direction[1]]

    def update(self, decision=0):
        if decision == 1:
            self.turn_right()
        elif decision == 2:
            self.turn_left()
        self.move()
        self.age += 1
        self.starve -= 1

    def grow(self):
        self.starve = 100
        self.body.append(self.old_tail)

    def turn_right(self):
        a, b = self.direction[0], self.direction[1]
        self.direction[0], self.direction[1] = b, -a

    def turn_left(self):
        a, b = self.direction[0], self.direction[1]
        self.direction[0], self.direction[1] = -b, a
