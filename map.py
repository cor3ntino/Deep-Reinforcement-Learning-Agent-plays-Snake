import random
import math
from constants import *
from numba import jit
import numpy as np

class Map:

    def __init__(self, snake):
        self.snake = snake
        self.structure = MAP
        self.add_food()
    
    def add_food(self):

        x, y = random.randint(1, GAME_DIMENSION - 2), random.randint(1, GAME_DIMENSION - 2)
        while [x, y] in self.snake.body:
            x, y = random.randint(1, GAME_DIMENSION - 2), random.randint(1, GAME_DIMENSION - 2)
        self.food = [x, y]

    def update(self):

        if self.snake.body[0] == self.food:   
            reward = 1               
            self.snake.grow()
            self.add_food()
        elif self.snake.body[0] in self.snake.body[1:]:
            reward = -1
            self.snake.alive = False
        elif self.structure[self.snake.body[0][0]][self.snake.body[0][1]] == 1:
            reward = -1
            self.snake.alive = False
        elif self.snake.starve < 1:
            reward = -0.01
            self.snake.alive = False
        else:
            reward = -0.01
        
        return(reward)

    def scan(self):
        
        def scan_wall(direction_x, direction_y, direction_range):
            res = 0
            for i in range(1, GAME_DIMENSION):                      
                if i < direction_range:
                    step_x = head_x + i * direction_x      
                    step_y = head_y + i * direction_y
                    if structure[step_x][step_y] == 1:   
                        res = 1 / distance((head_x, head_y), (step_x, step_y))
            return res
        
        def scan_self(direction_x, direction_y, direction_range):
            res = 0
            for i in range(1, GAME_DIMENSION):
                if i < direction_range:
                    step_x = head_x + i * direction_x
                    step_y = head_y + i * direction_y
                    if [step_x, step_y] in snake_body:
                        res = max(res, 1 / distance((head_x, head_y), (step_x, step_y)))
            return res

        def scan_food(direction_x, direction_y, direction_range):
            res = 0
            for i in range(1, direction_range):
                if food_x == (head_x + i * direction_x) and food_y == (head_y + i * direction_y):
                    res = 1
            return res
        
        scan = [0] * 21
        structure = self.structure
        snake_body = self.snake.body
        head_x = self.snake.body[0][0]
        head_y = self.snake.body[0][1]
        food_x = self.food[0]
        food_y = self.food[1]

        forward_x = self.snake.direction[0]         # calculating each coordinate for each 7 directions
        forward_y = self.snake.direction[1]         # since the snake sees in FIRST PERSON
        right_x = -forward_y
        right_y = forward_x
        left_x = forward_y                          # for example, if snake's looking in [1,0] direction (down)
        left_y = -forward_x                         # its left is [1,0] (right for us because we look from above)
        forward_right_x = forward_x + right_x
        forward_right_y = forward_y + right_y
        forward_left_x = forward_x + left_x
        forward_left_y = forward_y + left_y         # see snake.py class for better explanations
        backward_right_x = -forward_left_x
        backward_right_y = -forward_left_y
        backward_left_x = -forward_right_x
        backward_left_y = -forward_right_y

        forward_range = (GAME_DIMENSION - (forward_x * head_x + forward_y * head_y) - 1) % (GAME_DIMENSION - 1) + 1   # computing max range
        backward_range = (GAME_DIMENSION + 1) - forward_range                                             # for each direction
        right_range = (GAME_DIMENSION - (right_x * head_x + right_y * head_y) - 1) % (GAME_DIMENSION - 1) + 1
        left_range = (GAME_DIMENSION + 1) - right_range
        forward_right_range = min(forward_range, right_range)           # values are hard encoded
        forward_left_range = min(forward_range, left_range)             # since I'm not planning on making it modifiable
        backward_right_range = min(backward_range, right_range)
        backward_left_range = min(backward_range, left_range)

        scan[0] = scan_wall(forward_x, forward_y, forward_range)                 # scanning walls in all directions
        scan[1] = scan_wall(right_x, right_y, right_range)
        scan[2] = scan_wall(left_x, left_y, left_range)
        scan[3] = scan_wall(forward_right_x, forward_right_y, forward_right_range)
        scan[4] = scan_wall(forward_left_x, forward_left_y, forward_left_range)
        scan[5] = scan_wall(backward_right_x, backward_right_y, backward_right_range)
        scan[6] = scan_wall(backward_left_x, backward_left_y, backward_left_range)

        scan[7] = scan_food(forward_x, forward_y, forward_range)                 # scanning food in all directions
        scan[8] = scan_food(right_x, right_y, right_range)
        scan[9] = scan_food(left_x, left_y, left_range)
        scan[10] = scan_food(forward_right_x, forward_right_y, forward_right_range)
        scan[11] = scan_food(forward_left_x, forward_left_y, forward_left_range)
        scan[12] = scan_food(backward_right_x, backward_right_y, backward_right_range)
        scan[13] = scan_food(backward_left_x, backward_left_y, backward_left_range)

        scan[14] = scan_self(forward_x, forward_y, forward_range)                # scanning body in all directions
        scan[15] = scan_self(right_x, right_y, right_range)
        scan[16] = scan_self(left_x, left_y, left_range)
        scan[17] = scan_self(forward_right_x, forward_right_y, forward_right_range)
        scan[18] = scan_self(forward_left_x, forward_left_y, forward_left_range)
        scan[19] = scan_self(backward_right_x, backward_right_y, backward_right_range)
        scan[20] = scan_self(backward_left_x, backward_left_y, backward_left_range)

        vision = np.array(scan)
        return(vision)
    
    def give_matrice(self):
        
        matrice = np.array(MAP, dtype=float)

        matrice[self.food[1], self.food[0]] = 0.25

        matrice[self.snake.body[0][1], self.snake.body[0][0]] = 0.5

        for k in range(1, len(self.snake.body)):
            matrice[self.snake.body[k][1], self.snake.body[k][0]] = 0.75
        return(matrice)

@jit(nopython=True)
def distance(p1=None, p2=None):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    
