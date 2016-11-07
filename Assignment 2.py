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

Nx = 200
Ti = 0; Tf = 1.2
T = np.array([Ti,Tf])
c = 0.5

###################################################

def fW(V, dx, Nx ):
    Wx = np.zeros(len(V))    
    Wx[1:Nx+1] = (1./(2.*dx))*(V[2:Nx+2] - V[0:Nx])  
    Wx = Boundary(Wx , Nx)
    return Wx

def fV(W, dx, Nx):    
    Vx = np.zeros(len(W)) 
    Vx[1:Nx+1] = (1./(2.*dx))*(W[2:Nx+2] - W[0:Nx])  
    Vx = Boundary(Vx , Nx)    
    return Vx

def fU(V, dx, Nx ):    
    Ux = np.zeros(len(V))
    Ux[1:Nx+1] = V[1:Nx+1] 
    Ux = Boundary(Ux , Nx)    
    return Ux 

def Boundary(Ux, Nx):
    Ux[0] = Ux[Nx]
    Ux[Nx+1] = Ux[1]
    return Ux

###################################################

def Euler(Ui, Vi, Wi, dt, dx, Nx):
    # using Euler method
    k = fW(Vi, dx , Nx)
    W = Wi + k*dt    
    
    m = fV(Wi, dx , Nx)
    V = Vi + m*dt

    l = fU(Vi, dx , Nx)
    U = Ui + l*dt
    
    return U[1:Nx+1], V[1:Nx+1], W[1:Nx+1]

###################################################

def Rk4(Ui, Vi, Wi, dt, dx, Nx):
    # using Runge-Kutta method
    k1 = fW(Vi, dx, Nx)
    m1 = fV(Wi, dx, Nx)
    l1 = fU(Vi, dx, Nx)
        
    k2 = fW(Vi + m1*dt/2., dx, Nx)
    m2 = fV(Wi + k1*dt/2., dx, Nx)
    l2 = fU(Vi + l1*dt/2., dx, Nx)    

    k3 = fW(Vi + m2*dt/2., dx, Nx)
    m3 = fV(Wi + k2*dt/2., dx, Nx)
    l3 = fU(Vi + l2*dt/2., dx, Nx)
    
    k4 = fW(Vi + m3*dt, dx, Nx)  
    m4 = fV(Wi + k3*dt, dx, Nx)  
    l4 = fU(Vi + l3*dt, dx, Nx)  
  
    U_ = Ui + (dt/6.)*(l1 + 2.*l2 + 2.*l3 + l4)  # U value
    V_ = Vi + (dt/6.)*(m1 + 2.*m2 + 2.*m3 + m4)  # V value
    W_ = Wi + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)  # W value

    return U_[1:Nx+1], V_[1:Nx+1], W_[1:Nx+1]
        
#################################################        

def Func(x, sigma):

    return np.exp(-(x**2.)/(2.*sigma**2.))
    
################################################

def AdvSolver(Xc, Nx, T, c, Func, method):

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
    W[1:Nx+1,0] = -100.*x[1:Nx+1]*Xinit

    #boundary
    U[:,0] = Boundary(U[:,0], Nx); V[:,0] = Boundary(V[:,0], Nx); W[:,0] = Boundary(W[:,0], Nx)

    for i in range(1,Nt+1):	 	
        """THIS CODE MUST CHANGE TO DO EULER AND RK4"""
        U[1:Nx+1,i], V[1:Nx+1,i], W[1:Nx+1,i] = method(U[:,i-1], V[:,i-1], W[:,i-1], dt, dx, Nx)

        U[:,i] = Boundary(U[:,i], Nx); V[:,i] = Boundary(V[:,i], Nx); W[:,i] = Boundary(W[:,i], Nx)      

    return x, t, U

################################################

n = np.array([50,100,200])

order = 1

def L2norm(AdvSolver, Xc, n, T, c, Func, method, order): 
    """ Creating a self-convergence test to see for correct order """
    x1, t1, U1 = Solver(Xc, n[0], T, c, Func, method)  
    x2, t2, U2 = Solver(Xc, n[1], T, c, Func, method)  
    x4, t4, U4 = Solver(Xc, n[2], T, c, Func, method)  

    Udt1 = np.zeros(len(t1))
    Udt2 = np.zeros(len(t1))
    
    for i in range(len(t1)): 
        for j in range(1,len(x1)):
            Udt1[i] += (U1[j,i] - U2[2*j-1,2*i])**2
            Udt2[i] += (U2[2*j-1,2*i] - U4[4*j-3,4*i])**2
            
    diff1 = np.sqrt((1/n[0])*Udt1)                
    diff2 = (2**order)*np.sqrt((1/n[1])*Udt2)  

    return diff1, diff2, t1
    
#################################################    
    
x1, t1, U = AdvSolver(Xc, Nx, T, c, Func, Rk4)


diff1, diff2, t = L2norm(AdvSolver, Xc, n, T, c, Func, Euler, order)

line1, = plt.plot(x1, U[:,0], linewidth=2.0, color='r',label='re') 

for i in range(1,len(t1)): # this steps through t values
    line1.set_ydata(U[:,i]) # changes the data for line1 
    plt.draw()

#Nt = len(t1)/1.2
#
#plt.plot(x1, U[:,Nt*0.8])
#plt.plot(x1, U[:,Nt*0.9])
#plt.plot(x1, U[:,Nt*1.1])

plt.figure()
plt.plot(t,diff1/diff2)
plt.figure()
plt.plot(t,diff1); plt.plot(t,diff2)
plt.ioff() 

plt.show()
