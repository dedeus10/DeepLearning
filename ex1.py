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
#-- File		: ex1.py                  	                              	   --
#-- Authors     : Luis Felipe de Deus                                          --
#-- Professor   : Rodrigo da Silva Guerra                                      -- 
#--                                                                            -- 
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Created     : 11 Mar 2020                                                  --
#-- Update      : 12 Mar 2020                                                  --
#--------------------------------------------------------------------------------
#--                              Overview                                      --
#-- A simple example from numpy and math libraries                             --
#-- Code executed in python3                                                   --
#--------------------------------------------------------------------------------
#

#Import the libraries we need
from math import cos as cos1
from numpy import cos as cos2

# ----------------------------------------- MAIN ----------------------------------------------------------------------------------
#With math libraryaaa
arg = 1.5
a = cos1(arg)
print("Math result for cos(",arg,"):",a)
#With numpy library
b = cos2(arg)
print("Numpy result for cos(",arg,"):",b)
#difference between libraries
c = a - b
print("Difference:",c)
print("\n Não há diferença no resultado desta operação, ainda que a implementação tenha sido diferente. O Math é uma biblioteca basica/padrão do python, tendo como maior função trabalhar com dados escalares, enquanto o Numpy é um pacote que necessita ser instalado, e com um maior numero de funções para trabalhar com matrizes, listas e o que a algebra necessitar, em geral se usa mais a Numpy, pois possui mais funções, no entanto se apenas fosse necessario um calculo de cosseno a Math seria melhor indicada por ser padrão no python e mais leve  ")