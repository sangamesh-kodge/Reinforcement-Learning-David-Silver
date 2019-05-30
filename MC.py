#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:16:48 2019

@author: skodge
"""
import numpy as np 
from env import step
import random as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

 

# initialize hyperparameters 
Nzero=1000
maxepisode=10000000
groupsize=10000
N=np.zeros([10, 21,2])


#initialise the value function and display parameters
Q=np.zeros([10, 21,2])
Performance=np.zeros([int(maxepisode/groupsize)])
avgG=0

#training
for episode in range (maxepisode):
    #empty past episode history
    G=0
    episodehistory=[]
    episodelength=0
    
    
    #initial card twist
    nextstate=[0,0]
    [nextstate, reward, done]=step(nextstate)
    nextstate[0]=nextstate[0]-1
    nextstate[1]=nextstate[1]-1
    while not done:
       
        # selecting action Epsilon Greedy policy
        epsilon=Nzero/(Nzero+N[nextstate[0]][nextstate[1]][0]+N[nextstate[0]][nextstate[1]][1])
        if (rd.random()<=epsilon):
            action=rd.choice([0,1])
        else:
            action=np.argmax(Q[nextstate[0]][nextstate[1]])
        
        #update N(s,a) and episode history 
        N[nextstate[0]][nextstate[1]][action]=N[nextstate[0]][nextstate[1]][action]+1
        episodehistory.append([nextstate[0],nextstate[1],action])
        episodelength=episodelength+1
        
        # Stick or Twist / next enviroment step
        nextstate[0]=nextstate[0]+1
        nextstate[1]=nextstate[1]+1
        [nextstate, reward, done]=step(nextstate, action)
        nextstate[0]=nextstate[0]-1
        nextstate[1]=nextstate[1]-1
        
        # reward computation
        G=G+reward
    
    #updating the Q values for the MC run
    for i in range(episodelength):
        alpha=1/N[episodehistory[i][0]][episodehistory[i][1]][episodehistory[i][2]]
        Q[episodehistory[i][0]][episodehistory[i][1]][episodehistory[i][2]]=Q[episodehistory[i][0]][episodehistory[i][1]][episodehistory[i][2]]+ alpha*(G-Q[episodehistory[i][0]][episodehistory[i][1]][episodehistory[i][2]]) 
    
    #progress display every after every groupsize number of runs 
    avgG=avgG+G
    if ((episode+1)%groupsize==0):
        print ("Episode{} reward:{}".format(episode+1,avgG/groupsize)) 
        Performance[int((episode+1)/groupsize-1)]=avgG
        avgG=0
    
    
# ploting the V(s) surface
x=np.zeros([10,21])
y=np.zeros([10,21])
V=np.zeros([10,21])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(10):
    for j in range(21):
        y[i][j]=j+1
        x[i][j]=i+1
        V[i][j]=np.max(Q[i][j])

ax.plot_surface(x, y, V)
ax.set_xlabel('Dealer showing')
ax.set_ylabel('Player Sum')
ax.set_zlabel('Value Function')
plt.show()
    
#ploting the average reward obtained for each group
plt.plot(Performance) 
    
#Testing the Q function
avgG=0
for episode in range (10000):
    G=0
    [nextstate, reward, done]=step([0,0])
    nextstate[0]=nextstate[0]-1
    nextstate[1]=nextstate[1]-1
    while not done:
        action=np.argmax(Q[nextstate[0]][nextstate[1]])
        
        nextstate[0]=nextstate[0]+1
        nextstate[1]=nextstate[1]+1
        [nextstate, reward, done]=step(nextstate, action)
        nextstate[0]=nextstate[0]-1
        nextstate[1]=nextstate[1]-1
        G=G+reward
    avgG=avgG+G
print("Trained results averaged over 10000 episodes is ", avgG/10000 )


#saving the action value function for comparing with the TD algorithm as described in question 3
#np.save('MC_Qvalue.npy',Q)
