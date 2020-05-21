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
#-- File		: exemplo_perceptron.py                	                  	   --
#-- Authors     : Luis Felipe de Deus                                          --
#-- Professor   : Rodrigo da Silva Guerra                                      -- 
#--                                                                            -- 
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Created     : 22 Mar 2020                                                  --
#-- Update      : 22 Mar 2020                                                  --
#--------------------------------------------------------------------------------
#--                              Overview                                      --
#-- A simple example from a perceptron network                                 --
#-- Code executed in python3                                                   --
#--------------------------------------------------------------------------------
#

#Import the libraries we need
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
#endfunction

class Perceptron:
    def __init__(self, n_inputs=4, n_hidden=8, n_outputs=1):
        self.w_ih = np.random.random((n_hidden, n_inputs))
        self.w_ho = np.random.random((n_outputs, n_hidden))
        self.b_hid = np.random.random((n_hidden))
        self.b_out = np.random.random((n_outputs))

    def compute(self, inputs):
        self.s_hid = np.dot(self.w_ih, inputs) + self.b_hid
        self.z_hid = sigmoid(self.s_hid)
        self.s_out = np.dot(self.w_ho, self.z_hid) + self.b_out
        self.z_out = sigmoid(self.s_out)
        
        return self.z_out

# ----------------------------------------- MAIN ----------------------------------------------------------------------------------

p = Perceptron()
print(p.compute([1.0,2.0,3.0,4.0]))

'''#Make graphics
x = np.linspace(-5,5,50)
y = sigmoid(x)
plt.plot(x,y)
plt.show()'''
