#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:00:29 2022

@author: Stefano Carpin
EECS269 - Reinforcement Learning
Environment for lab 2
"""

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts




# class implementing the Gambler's problem in the RL book (Example 4.3, page 84)
class GamblerEnvironment(py_environment.PyEnvironment):
    
    # creates and instance setting the probability of winning (head). Defaults to 0.4 if no parameter is given
    def __init__(self,head_probability=0.4):
        if head_probability< 0 or head_probability > 1:
            raise ValueError("head_probability is a probability and must be between 0 and 1")
        else:
            self.head_probability = head_probability
                    
        # action is the amount of the stake (includes 0, even though it is not useful)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=50, name='action') # the rules of the game disallow stakes higher than 50
        # state is the current amount of money
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=1, maximum = 99, name='state')
        self._state = np.random.randint(1,100)  # randomly initialize state between 1 and 99
        self._episode_ended = False

    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec


    def _reset(self):
        self._state = np.random.randint(1,100)  # randomly reset state between 1 and 99
        self._episode_ended = False
        return ts.restart(self._state)

    # computes transition and reewards as per the rules of the game
    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode, so reset the env and start a new episode
            # a new episode.
            return self.reset()
        
        if action < 0 :
            raise ValueError("Stake cannot be negative")
            
        if action > min(self._state,100-self._state):
            raise ValueError("Stake value outside the allowed range")
            
        # flip the coin and update state with win/loss
        if np.random.uniform() < self.head_probability:
            self._state += action 
        else:
            self._state -= action
        
        reward = 0  # reward is always 0 except when the player wins
        
        if self._state <= 0:   #terminate episode with loss
            self._episode_ended = True   
            
        if self._state == 100:  #terminate episode with win
            self._episode_ended = True
            reward = 1

        if self._episode_ended:  # returns time_step of the appropriate type depending on whether episode ended or not
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)
        
# simple main function to test things out... 
if __name__ == "__main__":
    env = GamblerEnvironment()  # create an instance of the environemnt with standard success probability
    
    # sinteract with the environment with a random policy for a few times...
    for _ in range(5):
        print("starting new game with random policy...")
        step = env.reset()
        print("initial capital is {}".format(step.observation))
        cumulative_reward = 0
        while not step.is_last():
            current_state = step.observation
            stake = np.random.randint( 1,min(current_state,100-current_state)+1) # disallow useless stake of 0...
            print("betting {}".format(stake))
            step = env.step(stake)
            cumulative_reward += step.reward
            print("current capital is {}".format(step.observation))
        if step.reward == 0:
            print("lost game, got reward {}".format(cumulative_reward))
        else:
            print("won game, got reward {}".format(cumulative_reward))
    
