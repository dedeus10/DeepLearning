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


# @brief: This Class makes a Perceptron Neural Network with configurable params of layers and neurons
# @param: n_inputs is the number of inputs from the first layer ex. sensor, pixels, data
# @param: n_hidden is the number of neurons in each layer
# @param: n_outputs is the number of neuron in the output layer
# @return:  the array with the activation value for each neuron
class Perceptron:
    def __init__(self, n_inputs=3, n_hidden=4, n_outputs=4):
        #Pesos das camadas ocultas h1 e h2
        self.w_ih1 = np.random.random((n_hidden, n_inputs))
        print("\n-- Pesos Wh1 -- \n", self.w_ih1)
        self.w_ih2 = np.random.random((n_hidden, n_hidden))
        print("\n-- Pesos Wh2 -- \n", self.w_ih2)
        #Pesos da camada de saida
        self.w_ho = np.random.random((n_outputs, n_hidden))
        print("\n-- Pesos Who-- \n", self.w_ho)
        #Bias para todas as camadas
        self.b_hid1 = np.random.random((n_hidden))
        self.b_hid2 = np.random.random((n_hidden))
        self.b_out = np.random.random((n_outputs))

    def compute(self, inputs):
        #Computa saida da primeira hidden layer e aplica a função de ativação sigmoid
        self.s_hid1 = np.dot(self.w_ih1, inputs) + self.b_hid1
        self.z_hid1 = sigmoid(self.s_hid1)
        #Computa saida da segunda hidden layer e aplica a função de ativação sigmoid
        self.s_hid2 = np.dot(self.w_ih2, self.z_hid1) + self.b_hid2
        self.z_hid2 = sigmoid(self.s_hid2)
        #Computa output layer e aplica a função de ativação sigmoid
        self.s_out = np.dot(self.w_ho, self.z_hid2) + self.b_out
        self.z_out = sigmoid(self.s_out)
        
        return self.z_out

# ----------------------------------------- MAIN ----------------------------------------------------------------------------------

p = Perceptron()
print("\n-- Saidas dos Neuronios -- \n", p.compute([1.0,2.0,3.0]))
