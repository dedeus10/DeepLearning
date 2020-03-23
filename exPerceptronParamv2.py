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
#-- File		: exPerceptronParam.py                	                  	   --
#-- Authors     : Luis Felipe de Deus                                          --
#-- Professor   : Rodrigo da Silva Guerra                                      -- 
#--                                                                            -- 
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Created     : 23 Mar 2020                                                  --
#-- Update      : 23 Mar 2020                                                  --
#--------------------------------------------------------------------------------
#--                              Overview                                      --
#-- Example of a Perceptron neural network with layers and neurons configurable--
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
# @param: nHid_layers is the number of hidden layers in the network 
# @param: n_outputs is the number of neuron in the output layer
# @return:  the array with the activation value for each neuron
class Perceptron:
    def __init__(self, n_inputs=3, n_hidden=[3,2,3], nHid_layers=3, n_outputs=3):
        self.w_ih = []  #Pesos das camadas ocultas
        self.b_hid = [] #Bias das camadas ocultas
        self.s_hid = [] #Saida das camadas ocultas
        self.z_hid = [] #Ativação das camadas ocultas

        self.nHid_layers = nHid_layers

        #Pesos das camadas ocultas
        for i in range(nHid_layers):
            if(i==0):
                self.w_ih.append(np.random.random((n_hidden[i], n_inputs)))    #Primeiro em relacao as entradas
            else:
                self.w_ih.append(np.random.random((n_hidden[i], n_hidden[i-1])))
            print("\n-- Pesos Sinapticos Wh[%d] -- \n" %(i), self.w_ih[i])

        #Pesos da camada de saida
        self.w_ho = np.random.random((n_outputs, n_hidden[len(n_hidden)-1]))
        print("\n-- Pesos Sinapticos Wout-- \n", self.w_ho)

        #Bias para as camadas ocultas
        for i in range(nHid_layers):
            self.b_hid.append(np.random.random((n_hidden[i])))
            print("\n-- Bias bh[%d] -- \n" %(i), self.b_hid[i])
        #Bias da camada de saida
        self.b_out = np.random.random((n_outputs))
        print("\n-- Bias out-- \n", self.b_out)

    def compute(self, inputs=[1.0, 2.0, 3.0]):
        #Computa saida da primeira hidden layer e aplica a função de ativação sigmoid
        self.s_hid.append(np.dot(self.w_ih[0], inputs) + self.b_hid[0])
        self.z_hid.append(sigmoid(self.s_hid[0]))
      
        for i in range(self.nHid_layers-1):          
            self.s_hid.append(np.dot(self.w_ih[i+1], self.z_hid[i]) + self.b_hid[i+1])
            self.z_hid.append(sigmoid(self.s_hid[i+1]))          

        #Computa output layer e aplica a função de ativação sigmoid
        self.s_out = np.dot(self.w_ho, self.z_hid[self.nHid_layers-1]) + self.b_out
        self.z_out = sigmoid(self.s_out)
        
        return self.z_out

# ----------------------------------------- MAIN ----------------------------------------------------------------------------------
print("\n\n#### Rede Neural Perceptron Configuravel ###\n")
config = str(input("Usar parametros pre definidos (Y/N) ?: "))
if(config == 'Y' or config == 'y'):
    n_inputs = 3
    outputs = 3
    hidden_layers = 3
    neurons = [3,2,3]
    inputs = [1.0,2.0,3.0]
else:
    n_inputs = int(input("1 - Numero de entradas: "))
    inputs = []
    for i in range(n_inputs):
        inputs.append(int(input("  *Entrada[%d]: "%(i))))

    #print(inputs)    
    outputs = int(input("2 - Numero de saidas: "))
    hidden_layers = int(input("3 - Numero de Hidden Layers: "))
    neurons = []
    for i in range(hidden_layers):
        neurons.append(int(input("  *Numero de neuronios na Layer[%d]: "%(i))))

p = Perceptron(n_inputs=n_inputs, n_hidden=neurons, nHid_layers=hidden_layers, n_outputs=outputs)
print("\n\n### Architecture Perceptron ###")
print("*Inputs: ", inputs)
print("*N Outputs: ", outputs)
print("*N Hidden Layers: ", hidden_layers)
print("*Neurons/layer: ", neurons)
print("\n -- Saidas dos Neuronios -- \n", p.compute(inputs))
