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

Nzero=1000
maxepisode=10000000
groupsize=10000

Q=np.zeros([11, 22,2])
Nq=np.zeros([11, 22,2])
Nv=np.zeros([11, 22])
Performance=np.zeros([int(maxepisode/groupsize)])
avgG=0

#training
for episode in range (maxepisode):
    G=0
    episodehistory=[]
    episodelength=0
    nextstate=[0,0]
    [nextstate, reward, done]=step(nextstate)
    while not done:
        epsilon=Nzero/(Nzero+Nv[nextstate[0]][nextstate[1]])
        Nv[nextstate[0]][nextstate[1]]=Nv[nextstate[0]][nextstate[1]]+1
        if (rd.random()<=epsilon):
            action=rd.choice([0,1])
        else:
            action=np.argmax(Q[nextstate[0]][nextstate[1]])
        Nq[nextstate[0]][nextstate[1]][action]=Nq[nextstate[0]][nextstate[1]][action]+1
        episodehistory.append([nextstate[0],nextstate[1],action])
        episodelength=episodelength+1
        [nextstate, reward, done]=step(nextstate, action)
        G=G+reward
        
    for i in range(episodelength):
        alpha=1/Nq[episodehistory[i][0]][episodehistory[i][1]][episodehistory[i][2]]
        Q[episodehistory[i][0]][episodehistory[i][1]][episodehistory[i][2]]=Q[episodehistory[i][0]][episodehistory[i][1]][episodehistory[i][2]]+ alpha*(G-Q[episodehistory[i][0]][episodehistory[i][1]][episodehistory[i][2]])   
          
    avgG=avgG+G
    if ((episode+1)%groupsize==0):
        print ("Episode{} reward:{}".format(episode+1,avgG/groupsize)) 
        Performance[int((episode+1)/groupsize-1)]=avgG
        avgG=0 
    
    
# plot
x=np.zeros([10,21])
y=np.zeros([10,21])
V=np.zeros([10,21])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(10):
    for j in range(21):
        y[i][j]=j+1
        x[i][j]=i+1
        V[i][j]=np.max(Q[i+1][j+1])

ax.plot_surface(x, y, V)
ax.set_xlabel('Dealer showing')
ax.set_ylabel('Player Sum')
ax.set_zlabel('Value Function')
plt.show()
     
    
#Test
avgG=0
for episode in range (10000):
    G=0
    [nextstate, reward, done]=step([0,0])
    while not done:
        action=np.argmax(Q[nextstate[0]][nextstate[1]])
        [nextstate, reward, done]=step(nextstate, action)
        G=G+reward
    avgG=avgG+G
   
    
print("Trained results averaged over 10000 episodes is ", avgG/10000 )
plt.plot(Performance)

V=np.zeros([11,22])
for i in range(11):
    for j in range(22):
        V[i][j]=np.max(Q[i][j])
        

