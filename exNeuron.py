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
#-- File		: exNeuron.py                  	                              	   --
#-- Authors     : Luis Felipe de Deus                                          --
#-- Professor   : Rodrigo da Silva Guerra                                      -- 
#--                                                                            -- 
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Created     : 22 Mar 2020                                                  --
#-- Update      : 22 Mar 2020                                                  --
#--------------------------------------------------------------------------------
#--                              Overview                                      --
#-- A simple example from an neuron                                            --
#-- Code executed in python3                                                   --
#--------------------------------------------------------------------------------
#

#Import the libraries we need
import numpy as np
from matplotlib import pyplot as plt

#Função de ativação sigmoid
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
#endfunction

#Função de ativação Tangente Hiperbolica
def hyperbolicTangent(z):
    return np.tanh(z)
#endfunction

class Neuron:
    w = 0
    b = 0
    def __init__(self, input_size=3):   #Construtor
        self.w = [-1.0,0.0,1.0]     #Define os pesos
        self.b = -0.5               #Define o bias
    def compute(self, inputs):  #Função que computa a ativação do neuronio
        s = np.dot(self.w, inputs) + self.b
        z = hyperbolicTangent(s)

        return z

# ----------------------------------------- MAIN ----------------------------------------------------------------------------------

n = Neuron()
print("Ativação para entrada 1.0,2.0,3.0: ",n.compute([1.0,2.0,3.0]))
print("Ativação para entrada 1.0,20.0,3.0: ",n.compute([1.0,20.0,3.0]))

#Make graphics
x = np.linspace(-5,5,50)
#y = sigmoid(x)
y = hyperbolicTangent(x)
plt.plot(x,y)
plt.show()
