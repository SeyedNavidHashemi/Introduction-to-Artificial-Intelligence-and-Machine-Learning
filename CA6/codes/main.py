from snake import *
from utility import *
from cube import *

import pygame
import matplotlib.pyplot as plt
import numpy as np
from tkinter import messagebox
from snake import Snake
from snake_super import Super_Snake


def plot_reward(rewards, name:str):
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    y_title = 'Rewards of snake' + name
    plt.ylabel(y_title)

    plt.show()

def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))

    snake_1 = Snake((255, 0, 0), (15, 15), SNAKE_1_Q_TABLE)
    snake_2 = Snake((255, 255, 0), (5, 5), SNAKE_2_Q_TABLE)
    snake_1.addCube()
    snake_2.addCube()

    snack = Cube(randomSnack(ROWS, snake_1), color=(0, 255, 0))

    clock = pygame.time.Clock()

    reward_of_snake_1 = []
    reward_of_snake_2 = []
    while True:
        reward_1 = 0
        reward_2 = 0
    
        pygame.time.delay(2)
        clock.tick(100000000) 
        
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                if messagebox.askokcancel("Quit", "Do you want to save the Q-tables?"):
                    save(snake_1, snake_2)
                pygame.quit()
                plot_reward(reward_of_snake_1, '1')
                plot_reward(reward_of_snake_2, '2')
                exit()
                
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                np.save(SNAKE_1_Q_TABLE, snake_1.q_table)
                np.save(SNAKE_2_Q_TABLE, snake_2.q_table)
                pygame.time.delay(1000)

        state_1, new_state_1, action_1 = snake_1.move(snack, snake_2)
        state_2, new_state_2, action_2 = snake_2.move(snack, snake_1)

        snack, reward_1, win_1, win_2 = snake_1.calc_reward(snack, snake_2)
        snack, reward_2, win_2, win_1 = snake_2.calc_reward(snack, snake_1)

        reward_of_snake_1.append(reward_1)
        reward_of_snake_2.append(reward_2)
        snake_1.update_q_table(state_1, action_1, new_state_1, reward_1)
        snake_2.update_q_table(state_2, action_2, new_state_2, reward_2)
        
        redrawWindow(snake_1, snake_2, snack, win)


if __name__ == "__main__":
    main()
