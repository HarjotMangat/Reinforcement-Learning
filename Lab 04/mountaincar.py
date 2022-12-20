#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:00:29 2022

@author: Stefano Carpin
EECS269 - Reinforcement Learning
Environment for lab 4
Mimics the mountain car task in the textbook (example 10.1)
"""

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


FORWARD = 1
NEUTRAL = 0
BACKRWARD = -1



ACTION = ACTION = {-1:"BACWARD" , 0:"NEUTRAL" , 1:"FORWARD"}  # for printing


# class implementing the mountain world task in the Barto-Sutton book (example 10.1)
class MountainCar(py_environment.PyEnvironment):
    
    # creates and instance of the grid world
    def __init__(self):
                    
        # action is encoded as per the symbolic constants above
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=-1, maximum=1, name='action') 
        # state is (x,xdot), i.e., first component is position, and second component is velocity
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=-1.2, maximum = 0.7, name='state')
        self._state = np.zeros((2,))
        self._episode_ended = False

    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # when reset, the car starts with 0 velocity and random position in (-0.6,0.4)
    def _reset(self):
        self._state[0] = np.random.uniform(-0.6,-0.4)
        self._state[1] = 0
        self._episode_ended = False
        return ts.restart(self._state)

    # computes transition and rewards as per the state evolution equations provided in the textbook
    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode, so reset the env and start a new episode
            return self.reset()
        
        # first check that the action is valid
        if action  <-1 or action > 1:
            raise ValueError("Invalid action given")
        
        reward = -1
        
        # update state; order matters; see equations at page 245
        self._state[1] = self._state[1] + 0.001*action - 0.0025 * np.cos(3*self._state[0])  # update velocity, first
        self._state[0] = self._state[0] + self._state[1]  # then update update position
        
        
        # now clip state components, if needed
        # first clip position
        if self._state[0] < -1.2:  # hit left bound?
            self._state[0] = -1.2  # remain inside
            self._state[1] = 0    # reset speed to 0
        elif self._state[0] > 0.5: # reached goal position?
            reward = 0   # to end episode
        # then clip velocity
        if self._state[1] < -0.07:
            self._state[1] = -0.07
        elif self._state[1] > 0.07:
            self._state[1] = 0.07
            
        if reward ==0:   #reached goal location?
            self._episode_ended = True    # then terminate episode

        if self._episode_ended:  # returns time_step of the appropriate type depending on whether episode ended or not
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)
        
# simple main function to test things out... 
if __name__ == "__main__":
    env = MountainCar()  # create an instance of the environemnt 
    
    # sinteract with the environment with a random policy ...
    print("Exploring grid world with random policy for at most 100 steps...")
    step = env.reset()
    print("Initial position is {}".format(step.observation))
    cumulative_reward = 0
    nstep = 0
    while not step.is_last() and (nstep < 100):  
        move = np.random.randint(-1,2) # execute random action
        print("Executing {}".format(ACTION[move]))
        step = env.step(move)
        cumulative_reward += step.reward
        print("Current state is {}".format(step.observation))
        nstep += 1
    print("Ended exploration in location {}".format(step.observation))
    print("Final reward: {}".format(cumulative_reward))
    
