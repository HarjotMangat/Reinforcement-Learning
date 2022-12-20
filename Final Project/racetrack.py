#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Stefano Carpin
EECS269 - Reinforcement Learning
Environment for Final Project
"""

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


# symbolic constants for the actions available; note that not all actions are
# available in every state -- see functions below
UP = 0 
DOWN = 1
LEFT = 2
RIGHT = 3
UPFAST = 4
DOWNFAST = 5
LEFTFAST = 6
RIGHTFAST = 7


ACTION = ["UP","DOWN","LEFT","RIGHT", "UPFAST", "DOWNFAST", "LEFTFAST", "RIGHTFAST"]  # for printing

"""
A class representing a race track environment. An abstract car
can move around executing the actions defined above. 
"""
class RaceTrackEnvironment(py_environment.PyEnvironment):

    
    # creates and instance of the race track; no parameters needed
    def __init__(self):
        
        # if you change these constants you can scale the circuit; do not change
        self._lmargin = self._rmargin = 6
        self._hmargin = 12
        self._tmargin = self._bmargin = 5
        self._vmargin = 3
        self._finish_line = self._lmargin + int(self._hmargin/2)
        
        self._COLS = self._lmargin + self._rmargin + self._hmargin
        self._ROWS = self._tmargin + self._vmargin + self._bmargin
        
        
        # action is encoded as per the symbolic constants above
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=7, name='action') 
        # state is the current position on the grid (row,column); positions are 0 indexed
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=0, maximum = max(self._ROWS,self._COLS)-1, name='state')
        self._state = np.zeros((2,))
        self._episode_ended = False

    # standard PyEnvironment methods
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # when reset, the agent always goes back to default position
    # top part of the ring, on the finish line
    def _reset(self):
        self._state[0] = int(self._tmargin/2)
        self._state[1] = self._finish_line 
        self._episode_ended = False
        return ts.restart(self._state)

    # computes transition and rewards 
    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode, so reset the env and start a new episode
            # a new episode.
            return self.reset()

        # save previous state to determine reward
        old_state = np.array(self._state)
        # compute new state based on action and model
        self._state,_ = self.sample_new_state(self._state,action)
        
        if self._in_oil_leak(): # additional noise not accounted for by the model
            self._state[0] += np.random.choice(a=[1,2])
            self._state[1] += np.random.choice(a=[-1,0,1],p=[0.4,0.2,0.4])
        
        # determine reward
        if self._crashed(self._state):
            reward = -1
            self._episode_ended = True
        elif self._crossed_finish_line(old_state, self._state):
            reward = 1
        else:
            reward = 0
        
        if self._episode_ended:  # returns time_step of the appropriate type depending on whether episode ended or not
             return ts.termination(self._state, reward)
        else:
             return ts.transition(self._state, reward)

    # internal methods; do not use them!
    def _crossed_finish_line(self,old,new):
        return  (old[1] < self._finish_line and new[1] >= self._finish_line)
            
    def _crashed(self,stateN):
        return (stateN[0] < 0) or (stateN[1] < 0) or (stateN[0] >= self._ROWS ) or (stateN[1] >= self._COLS) \
            or ( (self._tmargin <= stateN[0] < self._tmargin + self._vmargin) and (self._lmargin <= stateN[1] < self._lmargin + self._hmargin)   )
    
    def _in_oil_leak(self):
        return (self._state[0] == 5) and ( 18 <= self._state[1] <= 20)

    def _zone(self,state):
        if state[0] < self._tmargin:
            if state[1] < self._lmargin:
                return 1
            elif state[1] < self._lmargin + self._hmargin:
                return 2
            else:
                return 3
        elif state[0] < self._tmargin + self._vmargin:
            if state[1] < self._lmargin:
                return 8
            else:
                return 4
        else:
            if state[1] < self._lmargin:
                return 7
            elif state[1] < self._lmargin + self._hmargin:
                return 6
            else:
                return 5

    # additional interface methods for public use
    
    # number of rows in the grid
    def get_num_rows(self): 
        return self._ROWS
    
    # number of columns in the grid
    def get_num_cols(self): 
        return self._COLS
    
    # returns a new state  and reward sampled from the distribution P(s',r|s,a)
    def sample_new_state(self,state,action):
        # first check that the action is valid for the given state
        if not self.valid_action(state,action):
            raise ValueError("Invalid action given")
        
        noise = np.random.choice(a=[-1,0,1],p=[0.25,0.5,0.25])
        
        new_state = np.array(state)
        
        if action == RIGHT:
            new_state[1] += 1
            new_state[0] += noise
        elif action == RIGHTFAST:
            new_state[1] += 2
            new_state[0] += noise
        elif action == LEFT:
            new_state[1] -= 1
            new_state[0] += noise
        elif action == LEFTFAST:
            new_state[1] -= 2
            new_state[0] += noise
        elif action == UP:
            new_state[0] -=1
            new_state[1] += noise
        elif action == UPFAST:
            new_state[0] -= 2
            new_state[1] += noise
        elif action == DOWN:
            new_state[0] +=1
            new_state[1] += noise
        else: # action must be down fast...
            new_state[0] += 2
            new_state[1] += noise
        
        zone = self._zone(state)   # clip noise to avoid moving backwards
        if zone == 1:
            new_state[0] = min(new_state[0],state[0])
        elif zone == 2:
            new_state[1] = max(new_state[1],state[1])
        elif zone == 3:
            new_state[0] = max(new_state[0],state[0])
            new_state[1] = max(new_state[1],self._lmargin + self._hmargin)
        elif zone == 4:
            new_state[0] = max(new_state[0],state[0])
        elif zone == 5:
            new_state[0] = max(new_state[0],state[0])
        elif zone == 6:
            new_state[1] = min(new_state[1],state[1])
        elif zone == 7:
            new_state[0] = min(new_state[0],state[0])
            new_state[1] = min(new_state[1],self._lmargin)
        else: # zone ==8:
            new_state[0] = min(new_state[0],state[0])
            
        # now compute reward
        if self._crashed(new_state):
            reward = -1
        elif self._crossed_finish_line(state,new_state):
            reward = 1
        else:
            reward = 0
        
        return new_state,reward
            
    
    # returns the actions allowed for the given state (essetially, for state 
    # s it returns the set A(s) represented as a list.
    # state must be given as as (row, column) and can be list, array, or tuple
    # state must be a valid state; unpredictable results occur if wrong states are returned
    def valid_actions(self,state):
        if state[0] < self._tmargin:  # top part of the circuit
            if state[1] < self._lmargin:   #zone 1
                return [UP,UPFAST, RIGHT,RIGHTFAST,LEFT,LEFTFAST]
            elif self._lmargin <= state[1] < self._lmargin+self._hmargin:  # zone 2
                return [UP,UPFAST, RIGHT,RIGHTFAST,DOWN,DOWNFAST]
            else:
                return [LEFT, LEFTFAST, RIGHT,RIGHTFAST,DOWN,DOWNFAST]   #zone 3
        elif self._tmargin <= state[0] < self._tmargin + self._vmargin:  # middle part
            if state[1] < self._lmargin:
                return [UP,UPFAST,LEFT,LEFTFAST,RIGHT,RIGHTFAST]  # zone 8
            else: 
                return [DOWN,DOWNFAST,LEFT,LEFTFAST,RIGHT,RIGHTFAST]  # zone 4
        else: # lower part
            if state[1] < self._lmargin:   #zone 7
                return [UP,UPFAST, RIGHT,RIGHTFAST,LEFT,LEFTFAST]
            elif self._lmargin <= state[1] < self._lmargin+self._hmargin:  # zone 6
                return [UP,UPFAST, LEFT,LEFTFAST,DOWN,DOWNFAST]
            else:
                return [LEFT, LEFTFAST, RIGHT,RIGHTFAST,DOWN,DOWNFAST]   #zone 5
                
    # determines if an action is valid for a state
    # action must be given as an integer
    # state must be given as as (row, column) and can be list, array, or tuple
    def valid_action(self,state,action):
        return action in self.valid_actions(state)
    
    # returns a list with all states; each state is represented as (row,col)
    # note: states are not "ordered"
    def all_states(self):
        all_states_list = []
        for i in range(self._ROWS):
            for j in range(self._COLS):
                if (i < self._tmargin) or (i >= self._tmargin + self._vmargin):
                    all_states_list.append((i,j))
                else:  # zone 8 and 4
                    if (j < self._lmargin) or ( j >= self._lmargin + self._hmargin):
                        all_states_list.append((i,j))
                
        return all_states_list


# simple main function to test things out... 
if __name__ == "__main__":
    env = RaceTrackEnvironment()  # create an instance of the environemnt 
    
    done = False
    state = env.reset().observation
    print("Starting random policy...")
    while not done:
        print("Current state:",state)
        actionset = env.valid_actions(state)
        action = np.random.choice(actionset)
        print("Executing action:",ACTION[action])
        timestep = env.step(action)
        state = timestep.observation
        if timestep.is_last():
            print("Crashed!")
            done = True
        elif timestep.reward ==1:
            print("Finished one lap! Quitting now")
            done = True
        
    
    

