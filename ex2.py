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
#-- File		: ex2.py                  	                              	   --
#-- Authors     : Luis Felipe de Deus                                          --
#-- Professor   : Rodrigo da Silva Guerra                                      -- 
#--                                                                            -- 
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Created     : 11 Mar 2020                                                  --
#-- Update      : 11 Mar 2020                                                  --
#--------------------------------------------------------------------------------
#--                              Overview                                      --
#-- A simple example from numpy and matplotlib libraries using matrix          --
#-- Code executed in python3                                                   --
#--------------------------------------------------------------------------------
#

#Import the libraries we need
import numpy as np
from matplotlib import pyplot as plt

# ----------------------------------------- MAIN ----------------------------------------------------------------------------------
a = [1,2,3,4,5]
b = 3
print(a*b)
print("\nAo multiplicar uma lista de tamanho X por um numero inteiro Y o python irá criar uma lista de tamanha x*y com os valores inciais repetidos \n\n")

row = 100
col = 5
a = np.zeros((row,col), dtype=np.float64)

for i in range(row):
    for j in range(col):
        a[i][j] = i*j
print(a)

#Make graphics
x = np.linspace(-5,5,100)
y = 1.0 / (1.0 + np.exp(-x))

plt.plot(x,y)
plt.show()
