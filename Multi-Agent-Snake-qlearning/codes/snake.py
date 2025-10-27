from cube import Cube
from constants import *
from utility import *

import random
import numpy as np


LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.95
EPSILON = 0.005

def check_existence_around(snake_head, self_body, other_snake_body):
    head_x, head_y = snake_head
    left = 0
    if(((head_x - 1, head_y) in list(map(lambda z: z.pos, self_body[1:]))) or ((head_x - 1, head_y) in list(map(lambda z: z.pos, other_snake_body)))):
        left = 1
    right = 0
    if(((head_x + 1, head_y) in list(map(lambda z: z.pos, self_body[1:]))) or ((head_x + 1, head_y) in list(map(lambda z: z.pos, other_snake_body)))):
        right = 1
    up = 0
    if(((head_x, head_y - 1) in list(map(lambda z: z.pos, self_body[1:]))) or ((head_x, head_y - 1) in list(map(lambda z: z.pos, other_snake_body)))):
        up = 1
    down = 0
    if(((head_x, head_y + 1) in list(map(lambda z: z.pos, self_body[1:]))) or ((head_x, head_y + 1) in list(map(lambda z: z.pos, other_snake_body)))):
        down = 1
    return right, left, up, down

def get_relative_distance(a, b):
    if (a - b) <= 0:
       return b - a
    return a - b

def make_state(my_pos, snack_pos, other_snake_pos, dirnx, dirny, self_body, other_snake_body):
  next_x = my_pos[0] + dirnx
  next_y = my_pos[1] + dirny
  
  out_of_bound = 0
  to_other = 0
  if next_x < 0 or next_x >= ROWS or next_y < 0 or next_y >= ROWS:
    out_of_bound = 1
    
  # Check if the next position overlaps with the other snake's body
  if ((next_x, next_y) in other_snake_body) or ((next_x + dirnx, next_y + dirny) in other_snake_body):
    to_other = 1

  head_x, head_y = my_pos
  snack_x, snack_y = snack_pos

  snack_dx = snack_x - head_x
  snack_dy = snack_y - head_y

  if snack_dx < 0:
    x_dir = 0
  elif snack_dx == 0:
    x_dir = 1
  else:
    x_dir = 2

  if snack_dy < 0:
    y_dir = 0
  elif snack_dy == 0:
    y_dir = 1
  else:
    y_dir = 2

  right, left, up, down = check_existence_around(my_pos, self_body, other_snake_body)
  right_wall = 0
  if(my_pos[0] >= ROWS - 1):
      right_wall = 1
  left_wall = 0
  if(my_pos[0] < 1):
      left_wall = 1
  down_wall = 0
  if(my_pos[1] >= ROWS - 1):
      down_wall = 1
  up_wall = 1
  if(my_pos[1] < 1):
      up_wall = 1


  state_number_15 = (x_dir + 1) \
             + 3 * (y_dir + 1) \
             + 9 * right_wall \
             + 18 * left_wall \
             + 36 * up_wall \
             + 72 * down_wall
    
  index_13 = ((x_dir + 1) * 12) + ((y_dir + 1) * 4) + (out_of_bound * 2) + to_other #13

  xdir_idx = x_dir + 1
  ydir_idx = y_dir + 1 
    
  index_16 = \
        xdir_idx + \
        3 * ydir_idx +\
        3*3 * right_wall +\
        3*3*3 * left_wall +\
        3*3*3*3 * up_wall +\
        3*3*3*3*3 * down_wall +\
        3*3*3*3*3*3 * right +\
        3*3*3*3*3*3*3 * left +\
        3*3*3*3*3*3*3*3 * up +\
        3*3*3*3*3*3*3*3*3 * down

  return index_16

class Snake:
    body = []
    turns = {}
    def __init__(self, color, pos, file_name=None):
        # pos is given as coordinates on the grid ex (1,5)
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        try:
            self.q_table = np.load(file_name)
        except:
            self.q_table = np.zeros((3*3*3*3*3*3*3*3*3*3, 4))

        self.recent_positions = []  # To track recent positions
        self.max_recent_positions = 20  # The length of the history to consider for oscillation detection

        self.lr = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.epsilon = EPSILON

    def get_optimal_policy(self, state):
        optimal_action = np.argmax(self.q_table[state])

        action_to_direction = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        current_direction = (self.dirnx, self.dirny)
        reverse_direction = (-current_direction[0], -current_direction[1])

        if action_to_direction[optimal_action] == reverse_direction:
            sorted_actions = np.argsort(self.q_table[state])[::-1]
            for action in sorted_actions:
                if action_to_direction[action] != reverse_direction:
                    optimal_action = action
                    break

        return optimal_action

    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)
        return action

    def update_q_table(self, state, action, next_state, reward):
      current_q_value = self.q_table[state][action]
      max_next_q_value = np.max(self.q_table[next_state])
      new_q_value = (1 - LEARNING_RATE) * current_q_value + (LEARNING_RATE * (reward + (DISCOUNT_FACTOR * max_next_q_value)))
      self.q_table[state][action] = new_q_value

    def move(self, snack, other_snake):
        state =  make_state(self.head.pos, snack.pos, other_snake.head.pos, self.dirnx, self.dirny, self.body, other_snake.body)
        action = self.make_action(state)
        if action == 0: # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1: # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2: # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3: # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        new_state = make_state(self.head.pos, snack.pos, other_snake.head.pos, self.dirnx, self.dirny, self.body, other_snake.body)
        return state, new_state, action



    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            return True
        return False

    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False

        if self.check_out_of_board():
            reward -= 5000
            win_other = True
            # print("Out of bounds reset")
            self.reset((random.randint(3, 18), random.randint(3, 18)))

        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward += 1000

        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            reward -= 5000
            win_other = True
            self.reset((random.randint(3, 18), random.randint(3, 18)))

        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):

            if self.head.pos != other_snake.head.pos:
                reward -= 3000
                win_other = True
                self.reset((random.randint(3, 18), random.randint(3, 18)))
            else:
                if len(self.body) > len(other_snake.body):
                    reward += 100
                    win_self = True
                else:
                    reward -=100
                    win_other = True
                    self.reset((random.randint(3, 18), random.randint(3, 18)))

        #check if around is anything and direction is the same
        right, left, up, down = check_existence_around(self.head.pos, self.body, other_snake.body)
        if(right and self.dirnx == 1):
            reward -= 50
        elif(right and self.dirnx != 1):
            pass
        
        if(left and self.dirnx == -1):
            reward -= 50
        elif(left and self.dirnx != -1):
            pass

        if(up and self.dirny == -1):
            reward -= 50
        elif(up and self.dirny != -1):
            pass

        if(down and self.dirny == 1):
            reward -= 50
        elif(down and self.dirny != 1):
            pass
                
        relative_pos_x = self.head.pos[0] - snack.pos[0]
        relative_pos_y = self.head.pos[1] - snack.pos[1]

        dot_product = relative_pos_x * self.dirnx + relative_pos_y * self.dirny

        if dot_product < 0:
           reward += 50
        else:
           reward -= 50
        return snack, reward, win_self, win_other

    
    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)

    def predict_future_positions(self, steps=3):
        future_positions = []
        x, y = self.head.pos
        dx, dy = self.dirnx, self.dirny

        for _ in range(steps):
            x += dx
            y += dy
            future_positions.append((x, y))

        return future_positions