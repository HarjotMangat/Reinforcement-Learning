This program was written in python3
This program uses the following imports:
	numpy
	tf_agents

When writing my program, I used tf_agents version 0.13.0

The program estimates the value function of the two states for the Recycling Robot example from chapter 3.
The estimation of the value function will be output as 'State zero average' and 'State one average' with state zero being high charge and state one being low charge.

The program runs for 1000 steps and uses alpha of 0.9 and beta of 0.4
