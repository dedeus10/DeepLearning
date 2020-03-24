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
#-- File		: exPerceptron.py                	                  	   --
#-- Authors     : Luis Felipe de Deus                                          --
#-- Professor   : Rodrigo da Silva Guerra                                      -- 
#--                                                                            -- 
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Created     : 22 Mar 2020                                                  --
#-- Update      : 23 Mar 2020                                                  --
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

def linear(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x

#Função do neuronio AND
def neuronAnd(a,b):
    s = a*1.0+b*1.0-1.0
    z = linear(s)
    return z

#Função do neuronio OR
def neuronOr(a,b):
    s = a*1.0+b*1.0
    z = linear(s)
    return z

#Função do neuronio NOT
def neuronNot(a):
    s = a*(-1.0)+1.0
    z = linear(s)
    return z

#Classe de um neuronio geral
class Neuron:
    def __init__(self, w0, w1, bias):
        self.w0 = w0
        self.w1 = w1
        self.bias = bias
    def compute(self, a, b):
        s = self.w0 * a + self.w1 * b + self.bias
        z = linear(s)
        return z

# ----------------------------------------- MAIN ----------------------------------------------------------------------------------


#print("\n-- Saidas do Neuronio AND -- \n",neuronAnd(1,0))
#print("\n-- Saidas do Neuronio OR -- \n",neuronOr(1,0))
#print("\n-- Saidas do Neuronio NOT-- \n",neuronNot(1))

neurAnd = Neuron(1.0, 1.0, -1.0)
neurOr = Neuron(1.0, 1.0, 0.0)
neurNot = Neuron(-1.0, 0.0, 1.0)

inputs = [(0,0), (0,1), (1,0), (1,1)]

for a,b in inputs:
    print('a = ',a,'b = ',b,'AND = ',\
         neurAnd.compute(a,b), 'OR = ',\
         neurOr.compute(a,b), 'NOT = ',\
         neurNot.compute(a,0))
    