#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 07:27:22 2019

@author: skodge
"""

import numpy as np 
from env import step
import random as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import sys

def Q(state, action, weights):
    feature=np.zeros([36,1])
    dealer=[]
    player=[]
    if(state[0]>=1 and state[0]<=4 ):
        dealer.append(0)
    if (state[0]>=4 and state[0]<=7):
        dealer.append(1)
    if (state[0]>=7 and state[0]<=10):
        dealer.append(2)
            
    if(state[1]>=1 and state[1]<=6):
        player.append(0)
    if (state[1]>=4 and state[1]<=9):
        player.append(1)
    if (state[1]>=7 and state[1]<=12):
        player.append(2)
    if(state[1]>=10 and state[1]<=15):
        player.append(3)
    if(state[1]>=13 and state[1]<=18):
        player.append(4)
    if(state[1]>=16 and state[1]<=21):
        player.append(5)
         
    for i in range(len(dealer)):
        for j in range(len(player) ):
            feature[18*action+3*player[j]+dealer[i],0]=1
    
    Value=np.dot(feature.T,weights)
    return Value[0][0],feature


    
    
     

#hyperparameters
alpha=0.01 
epsilon=0.05
maxepisode=10000000
groupsize=10000


Lambda=0
Performance=np.zeros([int(maxepisode/groupsize)])
avgG=0
weights=np.random.rand(36,1)
#oldweights=weights


for episode in range (maxepisode):
    G=[]
    episodehistory=[]
    episodelength=0
    
    nextstate=[0,0]
    [nextstate, reward, done]=step(nextstate)

    
    
    while not done:
        
        qa0,_=Q(nextstate,0,weights)
        qa1,_=Q(nextstate,1,weights)
        
        if(rd.random()<=epsilon):
            action=rd.choice([0,1])
            
        elif (qa0>qa1):
            action=0
        elif(qa0<qa1):
            action=1
        else:
            action=rd.choice([0,1])
            
    
            
        episodehistory.append([nextstate[0],nextstate[1],action])
        episodelength=episodelength+1   
        [nextstate, reward, done]=step(nextstate,action)
        G.append(reward)        
        
        if not done:
            qa0,_=Q(nextstate,0,weights)
            qa1,_=Q(nextstate,1,weights)
        
        r=0 
        mf=1-Lambda
        
        for i in range(episodelength-1,-1,-1):
            r=r+G[i]
            if not done:
                BS=mf*(r+max(qa0,qa1))
            else:
                BS=mf*(r)
            mf=mf*Lambda
            Qt,F=Q([episodehistory[i][0],episodehistory[i][1]],episodehistory[i][2],weights)
            dw= alpha*(BS-Qt)*F
            weights=weights+dw
            
    #progress display every after every groupsize number of runs 
    avgG=avgG+G[episodelength-1]
    if ((episode+1)%groupsize==0):
        Performance[int((episode+1)/groupsize-1)]=avgG
        print ("Episode{} Lambda {} reward:{}".format(episode+1,Lambda,Performance[int((episode+1)/groupsize-1)]/groupsize)) 
        avgG=0
        #oldweights= weights
            
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
        qa0,_=Q([i+1,j+1],0,weights)
        qa1,_=Q([i+1,j+1],1,weights)        
        V[i][j]=max(qa0,qa1)

ax.plot_surface(x, y, V)
ax.set_xlabel('Dealer showing')
ax.set_ylabel('Player Sum')
ax.set_zlabel('Value Function')
plt.show()

#ploting the average reward obtained for each group
fig = plt.figure()
plt.plot(Performance) 
plt.show()

        