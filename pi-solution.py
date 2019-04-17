#!/usr/bin/env python
# coding=utf-8
import numpy as np

"""
1: Procedure Policy_Iteration(S,A,P,R)
2:           Inputs
3:                     S is the set of all states
4:                     A is the set of all actions
5:                     P is state transition function specifying P(s'|s,a)
6:                     R is a reward function R(s,a,s')
7:           Output
8:                     optimal policy π
9:           Local
10:                     action array π[S]
11:                     Boolean variable noChange
12:                     real array V[S]
13:           set π arbitrarily
14:           repeat
15:                     noChange ←true
16:                     Solve V[s] = ∑s'∈S P(s'|s,π[s])(R(s,a,s')+γV[s'])
17:                     for each s∈S do
18:                               Let QBest=V[s]
19:                               for each a ∈A do
20:                                         Let Qsa=∑s'∈S P(s'|s,a)(R(s,a,s')+γV[s'])
21:                                         if (Qsa > QBest) then
22:                                                   π[s]←a
23:                                                   QBest ←Qsa
24:                                                   noChange ←false
25:           until noChange
26:           return π
"""

states = [0, 1, 2, 3, 4]
actions = [0, 1]
N_STATES = len(states)
N_ACTIONS = len(actions)
P = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # transition probability
R = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # rewards

P[0, 0, 1] = 1.0
P[1, 1, 2] = 1.0
P[2, 0, 3] = 1.0
P[3, 1, 4] = 1.0
P[4, 0, 4] = 1.0


R[0, 0, 1] = 1
R[1, 1, 2] = 10
R[2, 0, 3] = 100
R[3, 1, 4] = 1000
R[4, 0, 4] = 1.0


gamma = 0.75

# initialize policy and value arbitrarily
policy = [0 for s in range(N_STATES)]
V = np.zeros(N_STATES)

print "Initial policy", policy
# print V
# print P
# print R

is_value_changed = True
iterations = 0
while is_value_changed:
    is_value_changed = False
    iterations += 1
    # run value iteration for each state
    for s in range(N_STATES):
        V[s] = sum([P[s, policy[s], s1] * (R[s, policy[s], s1] + gamma*V[s1])
                    for s1 in range(N_STATES)])
        # print "Run for state", s

    for s in range(N_STATES):
        q_best = V[s]
        # print "State", s, "q_best", q_best
        for a in range(N_ACTIONS):
            q_sa = sum([P[s, a, s1] * (R[s, a, s1] + gamma * V[s1])
                        for s1 in range(N_STATES)])
            if q_sa > q_best:
                print "State", s, ": q_sa", q_sa, "q_best", q_best
                policy[s] = a
                q_best = q_sa
                is_value_changed = True

    print "Iterations:", iterations
    # print "Policy now", policy

print "Final policy"
print policy
print V
