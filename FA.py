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
import math

Qref=np.load('MC_Qvalue.npy')

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

def MSE(Qref, weights):
    Value=0
    for i in range(10):
        for j in range(21):
            qa0,_=Q([i+1,j+1],0,weights)
            qa1,_=Q([i+1,j+1],1,weights)        
            Value=Value+(qa0-Qref[i][j][0])**2+(qa1-Qref[i][j][1])**2
    Value=Value/(10*21*2)
    return math.sqrt(Value)
    
    
     

#hyperparameters
alpha=0.01 
epsilon=0.05
maxepisode=10000000
groupsize=10000

MSE_Q=np.zeros([11,int(maxepisode/groupsize)+1])
W=np.zeros([11,36,1])
P=np.zeros([11,int(maxepisode/groupsize)])

for L in range(11):
    Lambda=L*0.1
    Performance=np.zeros([int(maxepisode/groupsize)])
    avgG=0
    weights=np.random.rand(36,1)
    oldweights=weights
    
    MSE_Q[L][0]=MSE(Qref,weights)
    
    
    for episode in range (maxepisode):
        G=[]
        episodehistory=[]
        episodelength=0
        
        nextstate=[0,0]
        [nextstate, reward, done]=step(nextstate)
    
        
        
        while not done:
            
            qa0,_=Q(nextstate,0,weights)
            qa1,_=Q(nextstate,1,weights)
            
            if(rd.random()<=1-episode/maxepisode+epsilon):
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
                qa0,_=Q(nextstate,0,oldweights)
                qa1,_=Q(nextstate,1,oldweights)
            
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
            MSE_Q[L][int((episode+1)/groupsize-1)+1]=MSE(Qref,weights)
            print ("Episode{} Lambda {} reward:{}".format(episode+1,Lambda,Performance[int((episode+1)/groupsize-1)]/groupsize)) 
            avgG=0
            oldweights= weights
                
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
    
    W[L]=weights
    P[L]=Performance

fig = plt.figure()
for i in range(11):
    plt.plot(MSE_Q[i])
plt.show()
    
fig=plt.figure()
Y=[MSE_Q[0][100],MSE_Q[1][100],MSE_Q[2][100],MSE_Q[3][100],MSE_Q[4][100],MSE_Q[5][100],MSE_Q[6][100],MSE_Q[7][100],MSE_Q[8][100],MSE_Q[9][100],MSE_Q[10][100]]
X=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
plt.plot(X,Y)        