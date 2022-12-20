=====================================
* Harjot Mangat                     *
* EECS 269 - Reinforcement Learning *
* Lab 02 - Monte Carlo              *
* Lab02_exploring-starts.py         *
=====================================

This program was written in python3
This program uses the following imports:
	numpy
	matplotlib
	random
	tf_agents
	gamblerproblem

tf_agents version 0.13.0

The program implements the pseudocode from Chapter 5 for Monte Carlo with Exploring Starts to estimate an optimal policy.
The criterion for stopping the search was simply to run for 5,000,000 episodes(it will take a VERY LONG time to run) to approximate the 'loop forever' section of the pseudocode.
The program will put out a graph at the end with the final policy from the episodes.
