#!/usr/bin/env python
# coding: utf-8

# Harjot Mangat
# EECS 269 - Reinforcement Learning
# Lab 02 - Monte Carlo with Exploring Starts

from gamblerproblem import GamblerEnvironment

env = GamblerEnvironment()


import numpy as np
import random
import matplotlib.pyplot as plt
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

#Initialize Policy
def policyspace(n):
    a = min(n, 100-n)
    return a


policy = {}
for key in range(1, 100):
    p = []
    for action in range(1, 1+min(key,100-key)):
        p.append(action)
    policy[key] = p



#Initialize Q(S,A)
Qfunction = {}
for key in policy.keys():
    Qfunction[key] = {a: 0.0 for a in range(1, 1+min(key,100-key))}



#Initialize the returns list as an empty list
returns = {}



games_won = 0
games_lost = 0
for _ in range(5000000):
        G = 0
        step = env.reset()
        episode = []
        #print("initial capital is {}".format(step.observation))
        cumulative_reward = 0
        
        first_timestep = []
        first_timestep.append(step.observation)
        stake = np.random.randint( 1,min(step.observation,100-step.observation)+1)
        #print('first action was: ', stake)
        first_timestep.append(stake)
        step = env.step(stake)
        first_timestep.append(int(step.reward))
        episode.append(first_timestep)
        while not step.is_last():
            current_state = step.observation            
            timestep = []
            timestep.append(current_state)
            
            if type(policy[current_state]) == int:
                choice = policy[current_state]
                stake = choice
            else:
                choice = random.choice(policy[current_state])
                stake = choice
            #print("betting {}".format(stake))
            step = env.step(stake)
            timestep.append(stake)
            cumulative_reward += step.reward
            timestep.append(int(step.reward))
            #At this point, timestep should containt [current_state, stake, reward]
            episode.append(timestep)
            #print("current capital is {}".format(step.observation))
        if step.reward == 0:
            #print("lost game, got reward {}".format(cumulative_reward))
            games_lost += 1
        else:
            #print("won game, got reward {}".format(cumulative_reward))
            games_won += 1
            
        #Episode finished, updating the policy 
        
        #loop backwards through t = T-1, T-2, ..., 0
        for i in reversed(range(0,len(episode))):
            state, action, reward = episode[i]
            s_a_pair = (state,action)
            #Calculate G for the episode
            G += reward
            
            if not s_a_pair in [(x[0],x[1]) for x in episode[0:i]]:
                if returns.get(s_a_pair):
                    returns[s_a_pair].append(G)
                else:
                    returns[s_a_pair] = [G]
                
                Qfunction[state][action]= sum(returns[s_a_pair]) / len(returns[s_a_pair])
                
                #Calulating the argmax for Q(S,a)
                Q_list = list(map(lambda x: x[1], Qfunction[state].items()))
                
                max_arg = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
               
                update = random.choice(max_arg)
                policy[state] = update + 1
                
                
print('Total Games won: ', games_won)
print('Total Games lost: ', games_lost)

#Print out the Graph of the policy
data = policy
names = list(data.keys())
x = names
y = []

for i in range(len(data)):
    if type(data[i+1]) == int:
        y.append(data[i+1])
        
    else:
        temp = random.choice(data[i+1])
        y.append(temp)

plt.bar(x,y)
plt.title("Policy")
plt.xlabel("Capital")
plt.ylabel("Stake")
plt.show()



