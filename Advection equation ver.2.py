# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:53:53 2016

@author: Admin
"""
from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import time

from PyQt4.QtGui import QApplication # essential for Windows
plt.ion() # needed for interactive plotting

a = -1; b = 1
Xc = np.array([a,b])
Nx = 50
Ti = 0; Tf = 15
T = np.array([Ti,Tf])
c = 0.05

def U1(Ui, dx, Nx, v):

    return  (v/(2*dx))*(Ui[2:Nx+2] - Ui[0:Nx])


def Euler(Ui, U1, dt, dx, Nx, v):
    """ The Y array has two values so we need to split them up 
    when using euler"""
    fU = U1(Ui, dx, Nx, v)

    U = Ui[1:Nx+1] - dt*fU

    return U
    
def func(x, sigma):
    return np.exp(-(x**2)/(2*sigma**2))

def solver(Xc, Nx, T, c, func, method):
    
    Ti, Tf = T
    a, b = Xc
    dx = (b-a)/Nx
    dt = dx*c
    Nt = int((Tf-Ti)/dt)
    x = np.linspace(a-dx,b,Nx+2)
    t = np.linspace(Ti,Tf,Nt+1) 
    U = np.zeros((Nx+2, Nt+1))
    V = np.zeros((Nx+2, Nt+1))
    W = np.zeros((Nx+2, Nt+1))
    
    #U[x,0]
    Xinit = func(x[1:Nx+1],0.1)
    U[1:Nx+1,0] = Xinit

    #boundary
    U[0,0] = U[Nx,0]
    U[Nx+1,0] = U[1,0]
    	
    for i in range(1,Nt+1):	 	
        """THIS CODE MUST CHANGE TO DO EULER AND RK4"""
        U[1:Nx+1,i] = method(U[:,i-1], U1, dt, dx, Nx, c)
       
        U[0,i] = U[Nx,i]  
        U[Nx+1,i] = U[1,i]

    return x, t, U

x, t, U = solver(Xc, Nx, T, c, func, Euler)

line1, = plt.plot(x, U[:,0], linewidth=2.0, color='r',label='re') 

for i in range(1,len(t)): # this steps through t values
    line1.set_ydata(U[:,i]) # changes the data for line1 
    plt.draw()

plt.show()
plt.ioff() 

