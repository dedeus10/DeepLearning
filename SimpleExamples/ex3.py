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
#-- File		: ex3.py                  	                              	   --
#-- Authors     : Luis Felipe de Deus                                          --
#-- Professor   : Rodrigo da Silva Guerra                                      -- 
#--                                                                            -- 
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Created     : 11 Mar 2020                                                  --
#-- Update      : 11 Mar 2020                                                  --
#--------------------------------------------------------------------------------
#--                              Overview                                      --
#-- A simple example from OO python code                                       --
#-- Code executed in python3                                                   --
#--------------------------------------------------------------------------------
#

#Import the libraries we need
from numpy.random import randint, shuffle
from numpy import pi

class Quadrado:
    lado = 0
    def __init__(self,l):
        self.lado = l
    def area(self):
        return self.lado**2

class Triangulo:
    base = 0
    altura = 0
    def __init__(self,b,h):
        self.base = b
        self.altura = h
    def area(self):
        return self.base * self.altura / 2.0

class Circulo:
    raio = 0
    def __init__(self,r):
        self.raio = r
    def area(self):
        return pi*(self.raio**2)
# ----------------------------------------- MAIN ----------------------------------------------------------------------------------
    
objectList = []
sizeList = 60
for i in range(sizeList):
    objectList.append(Quadrado(randint(1,10)))
    objectList.append(Triangulo(randint(1,10), randint(1,10)))
    objectList.append(Circulo(randint(1,10)))

shuffle(objectList)
print("\n ----------------------")

totalArea = 0
for i in range(sizeList):
    area = objectList[i].area()
    print("Area do Objeto[%d]: " %(i), area)
    totalArea = area+totalArea

print("\n ----------------------")
print("Area total:", totalArea)
print("Area Media:", totalArea/sizeList)


