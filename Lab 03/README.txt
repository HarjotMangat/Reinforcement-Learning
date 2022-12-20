=====================================
* Harjot Mangat                     *
* EECS 269 - Reinforcement Learning *
* Lab 03 - SARSA & Q-learning       *
* sarsa.py & qlearning.py           *
=====================================

*This program was written in python3
*This program uses the following imports:
	numpy
	matplotlib
	gridenvironment

*tf_agents version 0.13.0

*The programs implement the pseudocode for SARSA & Q-learning to derive a policy to the gridworld environment given.

*The criterion for stopping the search was simply to run for 100 episodes. After trying different numbers for epsilon and the number of episdoes, it appears that 100 episodes is enough for both algoriths to converge to a solution.

*The programs will output the Q(S,A) at the end for for the policy. A graph will be saved to disk in a .png format of the reward vs episodes.

*This program requires no inputs, so it can be run from a terminal or command line with a call to the python interpreter and the name of the file. Such as "python sarsa.py"