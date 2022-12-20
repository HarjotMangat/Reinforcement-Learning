=====================================
* Harjot Mangat                     *
* EECS 269 - Reinforcement Learning *
* Lab 04 - Mountain Car             *
* Lab04_Mountain_Car.py             *
=====================================

*This program was written in python3
*This program uses the following imports:
	numpy
	matplotlib
	mountaincar
	tiles3

*tf_agents version 0.13.0

*The program implements the pseudocode for Episodic Semi-gradient Sarsa to interact with the mountain car environment given.

*The setting were the same as in the book for exampe 10.1, iht=IHT(4096), tiles(iht,8,[8*x/(0.5+1.2),8*xdot/(0.07+0.07)],[A]), 500 episodes, alpha = *userinput* /8.

*The program will output the graph from figure 10.2. A graph will be saved to disk in a .png format of the learning curve. The name of the graph will be something like 'Mountain_Car_Learning_Curve_alpha_xxxx.png', where the alpha value depends on what the user input.

*This program requires no commnad line inputs, so it can be run from a terminal or command line with a call to the python interpreter and the name of the file. Such as "python Lab04_Mountain_Car.py". 

*There is one input asked during the program, it is for the value of alpha to be used (a float between 0 and 1).