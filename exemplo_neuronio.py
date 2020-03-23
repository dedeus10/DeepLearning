#!/usr/bin/env python3

#
#--------------------------------------------------------------------------------
#--                                                                            --
#--                 Universidade Federal de Santa Maria                        --
#--                        Centro de Tecnologia                                --
#--                 Curso de Engenharia de Computação                          --
#--                          Deep Learning                                     --
#--                 Santa Maria - Rio Grande do Sul/BR                         --
#--                                                                            --
#--------------------------------------------------------------------------------
#--                                                                            --
#-- File		: ex5.py                  	                              	   --
#-- Authors     : Luis Felipe de Deus                                          --
#-- Professor   : Rodrigo da Silva Guerra                                      -- 
#--                                                                            -- 
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Created     : 14 Mar 2020                                                  --
#-- Update      : 14 Mar 2020                                                  --
#--------------------------------------------------------------------------------
#--                              Overview                                      --
#-- A simple example from sigmoid fuction                                      --
#-- Code executed in python3                                                   --
#--------------------------------------------------------------------------------
#

#Import the libraries we need
import numpy as np
from matplotlib import pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
#endfunction

class Neuron:
    w = 0
    b = 0
    def __init__(self, input_size=5):
        self.w = np.random.random(input_size)
        self.b = np.random.random()
    def compute(self, inputs):
        s = np.dot(self.w, inputs) + self.b
        z = sigmoid(s)

        return z

# ----------------------------------------- MAIN ----------------------------------------------------------------------------------

n = Neuron()
print(n.compute([1.0,2.0,3.0,4.0,5.0]))

'''#Make graphics
x = np.linspace(-5,5,50)
y = sigmoid(x)
plt.plot(x,y)
plt.show()'''
