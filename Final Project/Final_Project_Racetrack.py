#!/usr/bin/env python
# coding: utf-8


from racetrack import RaceTrackEnvironment
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
#from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


UP = 0 
DOWN = 1
LEFT = 2
RIGHT = 3
UPFAST = 4
DOWNFAST = 5
LEFTFAST = 6
RIGHTFAST = 7


ACTION = ["UP","DOWN","LEFT","RIGHT", "UPFAST", "DOWNFAST", "LEFTFAST", "RIGHTFAST"]  # for printing


#set alpha value
alpha = 0.5

#set epsilon to a small value
epsilon = .05

discount = .9

env = RaceTrackEnvironment()


ROWS = env.get_num_rows()
COLS = env.get_num_cols()

#setup Q(s,a) to 0
# set as Q[row][col][action]
Q = np.zeros((ROWS,COLS,8))
for i in range(ROWS):
    Q[i] = i
    for j in range(COLS):
        Q[i][j] = j
        for action in range(8):
            actionset = env.valid_actions((i,j))
            #print(actionset)
            if action in actionset:
                Q[i][j][action] = 0
            else:
                Q[i][j][action] = -2000
                


episode_reward = []


for _ in range(500):
    
    done = False
    cumulative_reward = 0
    State = np.array(env.reset().observation, dtype=int)
    observed_states = []
    observed_actions = []
    while not done:       
        # current S
        observed_states.append(np.array(State))

        #choose A from S
        actionset = env.valid_actions(State)
        if np.random.rand() < 1 - epsilon:

            choices = np.argwhere(Q[State[0]][State[1]] == np.amax(Q[State[0]][State[1]]))
            action = np.random.choice(choices.flatten())

        else:
            action = np.random.choice(actionset)

        observed_actions.append(action)

        #take action A and observe next state S' and reward R
        step = env.step(action)
        state_prime = np.array(step.observation, dtype=int)
        reward = step.reward
        cumulative_reward += reward

        #update our Q
        if state_prime[0] < ROWS and state_prime[1] < COLS:
            Q[State[0]][State[1]][action] += alpha * (reward + discount*np.max(Q[state_prime[0]][state_prime[1]]) - Q[State[0]][State[1]][action])
            State = state_prime
        else:
            Q[State[0]][State[1]][action] += alpha * (reward - Q[State[0]][State[1]][action])
        
        #looping min(length of observed_states or n) times for the planning stage
        for n in range(min(len(observed_states),25)):
            
            #pick previously visited state S
            randChoice = np.random.randint(low=0,high=len(observed_states))
            randS = tuple(observed_states[randChoice])

            #take action A (a random valid action for the state)
            randA = np.random.choice(env.valid_actions(randS))
            old_state = list(randS)
            #query the model for S' and R
            next_state, model_reward = env.sample_new_state(old_state,randA)
            
            #update Q
            if next_state[0] < ROWS and next_state[1] < COLS:
                Q[randS[0]][randS[1]][randA] += alpha * (model_reward + discount*np.max(Q[int(next_state[0])][int(next_state[1])]) - Q[randS[0]][randS[1]][randA])
               
            else:
                Q[randS[0],randS[1],randA] += alpha * (model_reward - Q[randS[0]][randS[1]][randA])
        #checking for the end of the episode        
        if step.is_last():
            print("Crashed!")
            observed_states.append(state_prime)
            done = True

        elif step.reward == 1:
            print("++++++++++++++++++++++++Finished one lap!!!!!++++++++++++++++++++++++++++++++")
            print("reward so far is:", cumulative_reward)
            
    #episode finished PRINT THE PATH TAKEN
    print("episode reward was: ", cumulative_reward)
    episode_reward.append(cumulative_reward)
    

#get_ipython().run_line_magic('matplotlib', 'notebook')

#print(len(observed_states))
#fig = plt.figure(figsize=(10,7))
#ax = plt.axes(xlim=(0,24),ylim=(0,12))

#rows= []
#cols=[]
#def init():
#    ax.xaxis.set_major_locator(MultipleLocator(1))
#    ax.yaxis.set_major_locator(MultipleLocator(1))
#    ax.grid(which='major')

#def animate(Frame):
    
#    global observed_states 

#    ax.set_title('Frame ' + str(Frame))
#    rows.append(12-(observed_states[Frame][0]))
#    cols.append(observed_states[Frame][1])
    
#    plt.scatter(cols,rows,color='black')
#    plt.plot(cols,rows,color='red')
    
#    return ax
    
#anim = FuncAnimation(fig,animate, init_func=init,repeat=True,frames=len(observed_states),interval=250)
#anim.save('scatter.gif', writer='pillow')

#plt.grid()
#plt.show()


#get_ipython().run_line_magic('matplotlib', 'notebook')
y = episode_reward

plt.plot(y)
plt.title("Episode Reward over Time(Q-Learning)")
plt.xlabel("Episodes")
plt.ylabel("Episode Reward")
plt.savefig('Dyna-Q_racetrack.png')
plt.show()
