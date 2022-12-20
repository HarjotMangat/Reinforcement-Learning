#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:00:29 2022

@author: Stefano Carpin
EECS269 - Reinforcement Learning
Environment for lab 3
"""

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


UP = 0 
DOWN = 1
LEFT = 2
RIGHT = 3

DIMENV = 10

ACTION = ["UP","DOWN","LEFT","RIGHT"]  # for printing


# class implementing a grid world environment with unknown dynamics similar to the windy gridworld
# (ex 6.5 in the Sutton Barto textbook)
# to keep things simple, the size of the grid is fixed at 10x10 (DIMENV)
class GridEnvironment(py_environment.PyEnvironment):
    
    # creates and instance of the grid world
    def __init__(self):
                    
        # action is encoded as per the symbolic constants above
        # 0 up; 1 down; 2 left; 3 right
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action') 
        # state is the current position on the grid (row,column); positions are 0 indexed
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=0, maximum = DIMENV-1, name='state')
        self._state = np.zeros((2,))
        self._episode_ended = False

    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # when reset, the agent always goes back to position (0,0)
    def _reset(self):
        self._state[0] = self._state[1] = 0
        self._episode_ended = False
        return ts.restart(self._state)

    # internal utility function -- ignore it
    def inregion(self,a,b,c,d):  
        if (a <= self._state[0] <= b) and (c <= self._state[1] <= d):
            return True
        else:
            return False


    # computes transition and rewards 
    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode, so reset the env and start a new episode
            # a new episode.
            return self.reset()
        
        # first check that the action is valid
        if action < 0 or action > 3:
            raise ValueError("Invalid action given")
        
        # move agent, avoiding to exit the grid
        if action == UP:
            self._state[0] = max(0,self._state[0]-1)
        if action == DOWN:
            self._state[0] = min(DIMENV-1,self._state[0]+1)
        if action == LEFT:
            self._state[1] = max(0,self._state[1]-1)
        if action == RIGHT:
            self._state[1] = min(DIMENV-1,self._state[1]+1)
        
        # now handle tricky cells
        if self.inregion(5, 5, 5, 5):  # black hole in the middle...
            self._state[0] = self._state[1] = 0  # go back to the beginning
        elif self.inregion(3,5,6,8): # windy region pushes up         
            self._state[0] -=2
        elif self.inregion(5,5,1,2):  # teleporting closer to the goal
            self._state[0] = 8
            self._state[1] = 5
        
        reward = -1  # reward is always -1 and the objective is to get to the goal as soon as possible (highest reward)
        
        if self._state[0] == DIMENV-1 and self._state[1] == DIMENV-1:   #t reached goal location?
            self._episode_ended = True    # then terminate episode

        if self._episode_ended:  # returns time_step of the appropriate type depending on whether episode ended or not
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)
        
# simple main function to test things out... 
if __name__ == "__main__":
    env = GridEnvironment()  # create an instance of the environemnt 
    
    # sinteract with the environment with a random policy ...
    print("Exploring grid world with random policy...")
    step = env.reset()
    print("Initial position is {}".format(step.observation))
    cumulative_reward = 0
    while not step.is_last():
        move = np.random.randint(0,4) # execute random action
        print("Executing {}".format(ACTION[move]))
        step = env.step(move)
        cumulative_reward += step.reward
        print("Current state is {}".format(step.observation))
    print("Ended exploration in location {}".format(step.observation))
    print("Final reward: {}".format(cumulative_reward))
    
