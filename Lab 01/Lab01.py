#!/usr/bin/env python
# coding: utf-8

# Harjot Mangat
# EECS 269 
# Lab 01
# A program that creates an environment to represent the Recycling Robot example from chapter 3
# and implements a simple controller to interact with the enviroment with a random policy.


#import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts



import tf_agents as tf_agents
tf_agents.__version__


print(tf.__version__)



class RecyclingRobot(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._state = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='state')
        #self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def state_spec(self):
        return self._state
    
    def _step(self, action):
        alpha = 0.9
        beta = 0.4
        
        #print('start of step')

        if self._episode_ended:
            
            #print('ended episode')
            #print('____---RESETING---___')
            return self.reset()
            #return ts.termination(np.array([self._state], dtype=np.int32), reward=-3)

    # two sets of states. 
    # high(0)= search(0), wait(1), recharge(2)
    # low(1)= search(0), wait(1), recharge(2)
        
        #print('got here too!')
        #print('state is : ', self._state)
        #print('action is: ', action)
        
        

        if self._state == 0:
            next_move_high = np.random.uniform(0,1)
            #print('value of next_move_high is: ', next_move_high)
            if action == 0:
            #reward += 1.5
                if next_move_high <= alpha:
                    self._state = 0
                    #print('set the state here')
                    reward =+ 1.5
                    
                else:
                    self._state = 1
                    reward =+ 1.5
                    
            elif action == 1:
                reward =+ 0.5
                self._state = 0
                
                
            elif action == 2:
                self._state = 0
                reward =+ 0
                
            
            else:
                raise ValueError('`action` should be 0, 1 or 2')
        
        
            
        
        elif self._state == 1:
            
            #print('got this far!')
            next_move_low = np.random.uniform(0,1)
            #print('value of next_move_low is: ', next_move_low)
            
            
            if action == 0:
                if next_move_low <= beta:
                    reward =+ 1.5
                    self._state = 0
                    
                else:
                    #reward =+ -3
                    #print('_____MADE IT TO TERMINATION____')
                    self._episode_ended = True
                
            elif action == 1:
                reward =+ 0.5
                self._state = 1
                
            elif action == 2:
                self._state = 0
                reward =+ 0
                
            else:
                raise ValueError('`action` should be 0, 1 or 2.')
        
        if self._episode_ended:
            #print('---GOT TO HERE---')
            return ts.termination(np.array([self._state], dtype=np.int32), reward=-3 )
        
        else:
            #print('got to transition?')
            #print('the value of reward is: ', reward)
            return ts.transition(np.array([self._state], dtype=np.int32), reward)
            

environment = RecyclingRobot()
utils.validate_py_environment(environment, episodes=10)

#search_action = np.array(0, dtype=np.int32)
#wait_action = np.array(1, dtype=np.int32)
#recharge_action = np.array(2, dtype=np.int32)

environment = RecyclingRobot()
time_step = environment.reset()
num_steps = 1000
transitions = []
reward = 0
state_zero_reward = 0
state_zero_count = 0
state_one_reward = 0
state_one_count = 0

cumulative_reward = time_step.reward

for _ in range(num_steps):
    
    action = np.random.randint(0,3)
    cumulative_reward += time_step.reward
    if environment.state_spec() == 0:
        state_zero_reward += time_step.reward
        state_zero_count += 1
    else:
        state_one_reward += time_step.reward
        state_one_count += 1
    #print(environment.state_spec())
    next_time_step = environment.step(action)
    time_step = next_time_step


#cumulative_reward += time_step.reward
print('Final Reward = ', cumulative_reward)
print('State zero count =', state_zero_count)
print('State zero reward = ', state_zero_reward)
print('State zero average =', state_zero_reward/state_zero_count)
print('State one count = ', state_one_count)
print('State one reward = ', state_one_reward)
print('State one avearge = ', state_one_reward/state_one_count)

