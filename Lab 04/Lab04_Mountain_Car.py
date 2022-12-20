#!/usr/bin/env python
# coding: utf-8

# Harjot Mangat
# EECS 269 - Reinforcement Learning
# Lab 04 - Mountain Car


from tiles3 import tiles, IHT
from mountaincar import MountainCar
import numpy as np
import matplotlib.pyplot as plt



FORWARD = 1
NEUTRAL = 0
BACKRWARD = -1

ACTION = ACTION = {-1:"BACWARD" , 0:"NEUTRAL" , 1:"FORWARD"}  # for printing


numTilings = 8
maxSize = 4096
weights = [0]*maxSize

print("please enter a value for alpha. (Between 0 and 1)")
alphaVal = input('->')

alpha = float(alphaVal)/numTilings
epsilon = 0.05

iht = IHT(maxSize)


#x(s,a)
def vectX(x, xdot,A):
    indices = tiles(iht, numTilings, [8*x/(0.5+1.2),8*xdot/(0.07+0.07)],[A])
    featureMap = [0]*maxSize
    for _ in indices:
        featureMap[_] = 1
    return featureMap


def Qfunc(x,xdot,A,weights):
    return np.matmul(vectX(x,xdot,A),weights)

def Grad(x,xdot,A):
    return vectX(x,xdot,A)

env = MountainCar()



stepsPerEp = []
for _ in range(500):   
    
    step = env.reset()
    #print("Starting episode")
    cumulative_reward = 0
    
    #state S
    state = np.array(step.observation, dtype=float)
    #print('state is ,',state)
    nstep = 0

    #pick A (epsilon greedy)
    if np.random.rand() > epsilon:
        #print('Qfunc gives',[Qfunc(state[0],state[1],action,weights) for action in range(-1,2)])
        action = np.argmax([Qfunc(state[0],state[1],action,weights) for action in range(-1,2)]) - 1
        #print("epsilon greedy chose the action A, ",action)
    else:
        action = np.random.randint(-1,2)
        #print("random int chose the action A, ",action)

    #print('picked initial action as, ',action)
    while not step.is_last():

        #take action A and observe R, S'
        step = env.step(action)
        nstep += 1
        cumulative_reward += step.reward
        reward = step.reward
        statePrime = np.array(step.observation, dtype=float)
        #print("Executed action, ",(ACTION[action]), 'got reward, ',reward," and S' is, ",statePrime)


        #if S' is terminal
        if step.is_last():
            #update weights
            change = alpha *(reward - Qfunc(state[0],state[1],action,weights))
            weights = weights + change *np.array(Grad(state[0],state[1],action))
            #print('updated weights on terminal state, ',state)

        else:
            #choose A' (epsilon greedy)
            if np.random.rand() > epsilon:
                actionPrime = np.argmax([Qfunc(statePrime[0],statePrime[1],action,weights) for action in range(-1,2)]) - 1
                #print("epsilon greedy chose the action A' ",actionPrime)
            else:
                actionPrime = np.random.randint(-1,2)
                #print("random int chose the action A' ", actionPrime)
                
            #print("chose A' to be, ",actionPrime)
            
            #update weights
            change = alpha *(reward + Qfunc(statePrime[0],statePrime[1],actionPrime,weights) - Qfunc(state[0],state[1],action,weights))
            weights = weights + change * np.array(Grad(state[0],state[1],action))
            #print("updated weights and adjusting S' to S, A' to A")
            
            #change S,A for next action
            state = statePrime
            action = actionPrime

    print("Final reward: {}".format(cumulative_reward))
    print("Episode lasted for ",nstep," steps")
    stepsPerEp.append(nstep)


y = stepsPerEp

plt.plot(y)
plt.title("Mountain Car Learning Curve\n Alpha = {} / {}".format(alphaVal,numTilings))
plt.xlabel('Episodes')
plt.ylabel('Steps per Episode')

plt.savefig('Mountain_Car_Learning_Curve_alpha_{}.png'.format(alpha))
plt.show()

