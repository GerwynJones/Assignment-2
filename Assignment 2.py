#!/usr/bin/env python
"""
Created on Mon Oct 24 11:53:53 2016

@author: C1331824
"""
from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt

from PyQt4.QtGui import QApplication # essential for Windows
plt.close('all')
plt.ion() # needed for interactive plotting

a = -1; b = 1
Xc = np.array([a,b])

Nx = 100
Ti = 0; Tf = 1.2
T = np.array([Ti,Tf])
c = 0.25

###################################################

def fW(V, dx, Nx ):
    Wx = np.zeros(len(V))    
    Wx[1:Nx+1] = (1/(2*dx))*(V[2:Nx+2] - V[0:Nx])  
    return Wx

def fV(W, dx, Nx):    
    Vx = np.zeros(len(W)) 
    Vx[1:Nx+1] = (1/(2*dx))*(W[2:Nx+2] - W[0:Nx])  
    return Vx

def fU(V, dx, Nx ):    
    Ux = np.zeros(len(V))
    Ux[1:Nx+1] = V[1:Nx+1] 
    return Ux 

def Boundary(Ux, Nx):
    Ux[0] = Ux[Nx]
    Ux[Nx+1] = Ux[1]
    return Ux

###################################################

def Euler(Ui, Vi, Wi, dt, dx, Nx):
    # using Euler method
    Wx = fW(Vi, dx , Nx)
    W = Wi - Wx*dt    
    
    Vx = fV(Wi, dx , Nx)
    V = Vi - Vx*dt

    Ux = fU(Vi, dx , Nx)
    U = Ui - Ux*dt
    
    return U, V, W

###################################################

def Rk4(Ui, Vi, Wi, dt, dx, Nx):
    # using Runge-Kutta method
    kU1 = fU(Vi, dx, Nx)
    kV1 = fV(Wi, dx, Nx)
    kW1 = fW(Vi, dx, Nx)
        
    kU2 = fU(Vi + kV1*dt/2., dx, Nx)
    kV2 = fV(Wi + kW1*dt/2., dx, Nx)
    kW2 = fW(Vi + kV1*dt/2., dx, Nx)    

    kU3 = fU(Vi + kV2*dt/2., dx, Nx)
    kV3 = fV(Wi + kW2*dt/2., dx, Nx)
    kW3 = fW(Vi + kV2*dt/2., dx, Nx)
    
    kU4 = fU(Vi + kV3*dt, dx, Nx)  
    kV4 = fV(Wi + kW3*dt, dx, Nx)  
    kW4 = fW(Vi + kV3*dt, dx, Nx)  
  
    U = Ui + (dt/6.)*(kU1 + 2*kU2 + 2*kU3 + kU4)  # U value
    V = Vi + (dt/6.)*(kV1 + 2*kV2 + 2*kV3 + kV4)  # V value
    W = Wi + (dt/6.)*(kW1 + 2*kW2 + 2*kW3 + kW4)  # W value

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
    U[:,0] = Boundary(U[:,0], Nx); V[:,0] = Boundary(V[:,0], Nx); W[:,0] = Boundary(W[:,0], Nx)

    for i in range(1,Nt+1):	 	
        """THIS CODE MUST CHANGE TO DO EULER AND RK4"""
        U[:,i], V[:,i], W[:,i] = method(U[:,i-1], V[:,i-1], W[:,i-1], dt, dx, Nx)

        U[:,i] = Boundary(U[:,i], Nx); V[:,i] = Boundary(V[:,i], Nx); W[:,i] = Boundary(W[:,i], Nx)      

    return x, t, U

################################################

n = np.array([50,100,200])

order = 1

def L2norm(Solver, Xc, n, T, c, Func, method, order): 
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

    return diff1, diff2, t1
    
#################################################    
    
x1, t1, U = Solver(Xc, Nx, T, c, Func, Rk4)


#diff1, diff2, t = L2norm(Solver, Xc, n, T, c, Func, Rk4, order)
#
#line1, = plt.plot(x1, U[:,0], linewidth=2.0, color='r',label='re') 
#
#for i in range(1,len(t1)): # this steps through t values
#    line1.set_ydata(U[:,i]) # changes the data for line1 
#    plt.draw()

Nt = len(t1)/1.2
plt.plot(x1, U[:,Nt*0.9])
plt.plot(x1, U[:,Nt*1])
plt.plot(x1, U[:,Nt*1.1])

plt.figure()
plt.semilogy(t,diff1/diff2)
plt.figure()
plt.semilogy(t,diff1); plt.semilogy(t,diff2)
plt.ioff() 


