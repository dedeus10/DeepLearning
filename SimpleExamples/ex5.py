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

# ----------------------------------------- MAIN ----------------------------------------------------------------------------------
  
#Make graphics
x = np.linspace(-5,5,100)
y = sigmoid(x)
plt.plot(x,y)
plt.show()
