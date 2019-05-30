#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:02:11 2019

@author: skodge
"""

import numpy as np 
import random as rd

def step(state,a=0):
    blackcard=np.array(range(1,11))
    redcard=np.array(range(-10,0))
    nextstate=state
    
    if state==[0,0] :
        nextstate= [rd.choice(blackcard),rd.choice(blackcard)]
        reward=0
        done=False
        return nextstate,reward,done
    
    if a==1 :
        if (rd.random()<=2/3):
            nextstate[1]=nextstate[1]+rd.choice(blackcard)
            if(nextstate[1]>21):
                done =True
                reward=-1
            else:
                done=False
                reward=0
            return nextstate,reward,done
        else:
            nextstate[1]=nextstate[1]+rd.choice(redcard)
            if(nextstate[1]<1):
                done =True
                reward=-1
            else:
                done=False
                reward=0
            return nextstate,reward,done
    else:
        reward=0
        done= False
        
        while not done :    
            if (rd.random()<=2/3):
                nextstate[0]=nextstate[0]+rd.choice(blackcard)
                if(nextstate[0]>21):
                    done =True
                    reward=1
                    return nextstate,reward,done
                    
                else:
                    if (nextstate[0]>16):
                        done=True
                        if nextstate[0]>nextstate[1]:
                            reward=-1
                        else:
                            if (nextstate[0]<nextstate[1]):
                                reward =1
                            else: 
                                reward=0
                                
                        return nextstate,reward,done
                
            else:
                nextstate[0]=nextstate[0]+rd.choice(redcard)
                if(nextstate[0]<1):
                    done =True
                    reward=1
                    return nextstate,reward,done
                    
               
            
                
            
        
    