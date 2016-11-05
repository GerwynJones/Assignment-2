
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:53:53 2016

@author: C1331824
"""
from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import time

from PyQt4.QtGui import QApplication # essential for Windows
plt.ion() # needed for interactive plotting

a = -1; b = 1
Xc = np.array([a,b])

Nx = 100
Ti = 0; Tf = 1.2
T = np.array([Ti,Tf])
c = 0.5

###################################################


def rhs1(V, dx, Nx ):
    Wx = np.zeros(len(V))    
    Wx[1:Nx+1] = (1./(2.*dx))*(V[2:Nx+2] - V[0:Nx])  
    Wx = Boundary(Wx,Nx)
    return Wx

def rhs2(W, dx, Nx):    
    Vx = np.zeros(len(W)) 
    Vx[1:Nx+1] = (1./(2.*dx))*(W[2:Nx+2] - W[0:Nx])  
    Vx = Boundary(Vx,Nx)
    return Vx

def rhs3(V, dx, Nx ):    
    Ut = np.zeros(len(V))
    Ut[1:Nx+1] = V[1:Nx+1] 
    Ut = Boundary(Ut,Nx)
    return Ut 

def Boundary(Ut , Nx):
    Ut[0] = Ut[Nx]
    Ut[1] = Ut[Nx+1]
    return Ut

###################################################

def Euler(Ui, Vi, Wi, dt, dx, Nx, v):
    # using Euler method
   
    Wx = rhs1(Vi, dx , Nx)
    Wx[1:Nx+1] = Wi[1:Nx+1] + Wx[1:Nx+1]*dt    
    
    Vx = rhs2(Wi, dx , Nx)
    Vx[1:Nx+1] = Vi[1:Nx+1] + Vx[1:Nx+1]*dt

    Ux = rhs3(Vi, dx , Nx)
    Ux[1:Nx+1] = Ui[1:Nx+1] + Ux[1:Nx+1]*dt

    return Ux, Vx, Wx

###################################################

def Rk4(Ui, Vi, Wi, dt, dx, Nx, v):
    # using Runge-Kutta method
    kU1 = U1(Vi, dx, Nx)
    kV1 = V1(Wi, dx, Nx)
    kW1 = W1(Vi, dx, Nx)
        
    kU2 = U1(Vi + kU1*dt/2, dx, Nx)
    kV2 = V1(Wi + kV1*dt/2, dx, Nx)
    kW2 = W1(Vi + kW1*dt/2, dx, Nx)    

    kU3 = U1(Vi + kU2*dt/2, dx, Nx)
    kV3 = V1(Wi + kV2*dt/2, dx, Nx)
    kW3 = W1(Vi + kW2*dt/2, dx, Nx)
    
    kU4 = U1(Vi + kU3*dt, dx, Nx)  
    kV4 = V1(Wi + kV3*dt, dx, Nx)  
    kW4 = W1(Vi + kW3*dt, dx, Nx)  
  
    U = Ui[1:Nx+1] + (dt/6)*(kU1[1:Nx+1] + 2*kU2[1:Nx+1] + 2*kU3[1:Nx+1] + kU4[1:Nx+1])  # U value
    V = Vi[1:Nx+1] + (dt/6)*(kV1[1:Nx+1] + 2*kV2[1:Nx+1] + 2*kV3[1:Nx+1] + kV4[1:Nx+1])  # V value
    W = Wi[1:Nx+1] + (dt/6)*(kW1[1:Nx+1] + 2*kW2[1:Nx+1] + 2*kW3[1:Nx+1] + kW4[1:Nx+1])  # W value

    return U, V, W
        
#################################################        

def Func(x, sigma):

    return np.exp(-(x**2)/(2.*sigma**2))
    
################################################

def Solver(Xc, Nx, T, c, Func, method):

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
    Xinit = Func(x[1:Nx+1],0.1)
    U[1:Nx+1,0] = Xinit
    W[1:Nx+1,0] = -100*x[1:Nx+1]*Xinit

    #boundary
    U[:,0] = Boundary(U[:,0], Nx)
    V[:,0] = Boundary(V[:,0], Nx)
    W[:,0] = Boundary(W[:,0], Nx)   
    
    for i in range(1,Nt+1):	 	
        """THIS CODE MUST CHANGE TO DO EULER AND RK4"""
        U[:,i], V[:,i], W[:,i] = method(U[:,i-1], V[:,i-1], W[:,i-1], dt, dx, Nx, c)

        U[:,i] = Boundary(U[:,i], Nx)  
        V[:,i] = Boundary(V[:,i], Nx)
        W[:,i] = Boundary(W[:,i], Nx)

    return x, t, U

################################################

n = np.array([Nx,2*Nx,4*Nx])

order = 1

def Convergence(Solver, Xc, n, T, c, Func, method, order): 
    """ Creating a self-convergence test to see for correct order """
    x1, t1, U1 = Solver(Xc, n[0], T, c, Func, method)  
    x2, t2, U2 = Solver(Xc, n[1], T, c, Func, method)  
    x4, t4, U4 = Solver(Xc, n[2], T, c, Func, method)  

    Udt1 = np.zeros(len(t1))
    Udt2 = np.zeros(len(t1))
    
    for i in range(len(t1)): 
    
        Udt1[i] = np.sum((U1[:-1,i] - U2[::2,2*i])**2)
        Udt2[i] = np.sum((U2[::2,2*i] - U4[::4,4*i])**2)
            
    diff1 = np.sqrt((1/n[0])*Udt1)                
    diff2 = (2**order)*np.sqrt((1/n[0])*Udt2)  

    return diff1, diff2
    
#################################################    
    
x, t, U = Solver(Xc, Nx, T, c, Func, Euler)

diff1, diff2 = Convergence(Solver, Xc, n, T, c, Func, Euler, order)

line1, = plt.plot(x, U[:,0], linewidth=2.0, color='r',label='re') 

for i in range(1,len(t)): # this steps through t values
    line1.set_ydata(U[:,i]) # changes the data for line1 
    plt.draw()


plt.figure()
plt.plot(t,diff1/diff2)
plt.figure()
plt.plot(t,diff1); plt.plot(t,diff2)
plt.ioff() 




