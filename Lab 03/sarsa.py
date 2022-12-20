#!/usr/bin/env python
# coding: utf-8


from gridenvironment import GridEnvironment
import numpy as np
import matplotlib.pyplot as plt

env = GridEnvironment()

UP = 0 
DOWN = 1
LEFT = 2
RIGHT = 3

DIMENV = 10

ACTION = ["UP","DOWN","LEFT","RIGHT"]


#set alpha to 0.5
alpha = 0.5

#set epsilon to a small value
epsilon = .05


#set up Q(S,A) to 0
# set as Q[state][state][action]
Q = np.zeros((10,10,4))
for i in range(10):
    Q[i] = i
    for j in range(10):
        Q[i][j] = j
        for action in range(4):
            Q[i][j][action] = 0


#keep track of the total reward per episode
episode_reward = []


#looping through episodes
for _ in range(100):
    
    #initialized state to 0,0
    step = env.reset()
    #print("Initial position is {}".format(step.observation))
    cumulative_reward = 0
    state = np.array(step.observation, dtype=int)
    
    #choose initial A from S using epsilon greedy
    if np.random.rand() < 1 - epsilon:
        move = np.argmax(Q[state[0]] [state[1]])
    else:
        move = np.random.choice(len(Q[state[0]][state[1]]))
    
    
    #loops an entire episode, one step at a time
    while not step.is_last():
        
        #print("Executing action a {}".format(ACTION[move]))
		
        #observe reward and S'
        step = env.step(move)
        #print("Current state is {}".format(step.observation))
        state_prime = np.array(step.observation, dtype=int)
        reward = step.reward
        cumulative_reward += step.reward
        
        #choose A' from S'
        if np.random.rand() < 1 - epsilon:
            move_prime = np.argmax(Q[int(step.observation[0])] [int(step.observation[1])])
        else:
            move_prime = np.random.choice(len(Q[int(step.observation[0])][int(step.observation[1])]))
        
        Q[state[0]][state[1]][move] += alpha * (reward + Q[state_prime[0]][state_prime[1]][move_prime] - Q[state[0]][state[1]][move])
        
        state = state_prime
        move = move_prime
        
    #print("Ended exploration in location {}".format(step.observation))
    #print("Final reward: {}".format(cumulative_reward))
    episode_reward.append(cumulative_reward)


for i in range(10):
    for j in range(10):
            print("({},{})/{}".format(i,j,ACTION[np.argmax(Q[i][j])]))



y = episode_reward

plt.plot(y)
plt.title("Episode Reward over Time(SARSA)")
plt.xlabel("Episodes")
plt.ylabel("Episode Reward")

plt.savefig('Sarsa.png')
#plt.show()

