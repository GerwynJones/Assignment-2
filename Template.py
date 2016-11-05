# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 10:27:20 2016

@author: gezer
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
#=======================================

a = -1; b = 1
Xc = np.array([a,b])

Nx = 100
Ti = 0; Tf = 1.2
T = np.array([Ti,Tf])
c = 0.5

## This solves a 1D wave equation to see if the method works.

def rhs1(V, dx, Nx ):
    Wt = np.zeros(len(V))    
    Wt[1:Nx+1] = (1./(2.*dx))*(V[2:Nx+2] - V[0:Nx])  
    Wt = boundary(Wt,Nx)
    return Wt

def rhs2(W, dx, Nx):    
    Vt = np.zeros(len(W)) 
    Vt[1:Nx+1] = (1./(2.*dx))*(W[2:Nx+2] - W[0:Nx])  
    Vt = boundary(Vt,Nx)
    return Vt

def rhs3(V, dx, Nx ):    
    Ut = np.zeros(len(V))
    Ut[1:Nx+1] = V[1:Nx+1] 
    Ut = boundary(Ut,Nx)
    return Ut 
    
def boundary(Ut , Nx):
    Ut[0] = Ut[Nx]
    Ut[1] = Ut[Nx+1]
    return Ut

def Func(x , sigma): ## Sets initial conditions at time t = 0
    u = np.exp(-(x**2)/(2.*sigma**2))
    v = 0
    w = -100*x*u
    return u, v, w

def Euler( Ui , Vi, Wi, dx , dt , Nx ):
   
    Wx = rhs1(Vi, dx , Nx )
    Wx[1:Nx+1] = Wi[1:Nx+1] + Wx[1:Nx+1]*dt    
    
    Vx = rhs2(Wi, dx , Nx )
    Vx[1:Nx+1] = Vi[1:Nx+1] + Vx[1:Nx+1]*dt

    Ux = rhs3(Vi, dx , Nx )
    Ux[1:Nx+1] = Ui[1:Nx+1] + Ux[1:Nx+1]*dt

    return Ux, Vx, Wx



def AdvSolve(Xc, Nx, T, c, Func, method): ## v for velocity.
    
    Ti, Tf = T
    a, b = Xc
    dx = (b-a)/Nx
    dt = dx*c
    Nt = int((Tf-Ti)/dt)
    xgrid = np.linspace(a-dx,b,Nx+2)
    tgrid = np.linspace(Ti,Tf,Nt+1) 
    
    U = np.zeros((Nx + 2, Nt + 1))
    V = np.zeros((Nx + 2 , Nt + 1))
    W = np.zeros((Nx + 2 , Nt + 1))
    
    ## initial conditions.
    V[1:Nx + 1, 0] = Func(xgrid[1:Nx+1] , 0.1)[1]
    W[1:Nx + 1, 0] = Func(xgrid[1:Nx+1] , 0.1)[2]
    U[1:Nx + 1, 0] = Func(xgrid[1:Nx+1] , 0.1)[0] ## sigma is 0.1 here.
    
    
    ## Boundary conditions
    
    U[:, 0] = boundary(U[:, 0] , Nx)
    V[:, 0] = boundary(V[:, 0] , Nx)
    W[:, 0] = boundary(W[:, 0] , Nx)
    
    for t in range(1 , int(Nt + 1)):
        
        U[:, t] = method(U[:, t-1], V[:, t-1], W[:, t-1], dx , dt , Nx )[0]           
        V[:, t] = method(V[:, t-1], V[:, t-1], W[:, t-1], dx , dt , Nx )[1]           
        W[:, t] = method(W[:, t-1], V[:, t-1], W[:, t-1], dx , dt , Nx )[2]           

    return U, xgrid


U, x = AdvSolve(Xc, Nx, T, c, Func, Euler)
U2, x2 = AdvSolve(Xc, Nx, T, c, Func, Euler)
U3, x3 = AdvSolve(Xc, Nx, T, c, Func, Euler)

plt.figure()
plt.plot(x,U[:,0])
plt.plot(x,U[:,26])
plt.plot(x,U[:,51])


#figure()   
#semilogy(tgridE100, l[:-1])
#semilogy(tgridE200, 2*l2[:-1])



def RK4(U, V, W, dx, dt, Nx):  
        
    n1=rhs1(V, dx, Nx)
    m1=rhs2(W, dx, Nx)
    k1=rhs3(V, dx, Nx)
    n2=rhs1(V+0.5*m1*dt, dx, Nx)
    m2=rhs2(W+0.5*n1*dt, dx, Nx)
    k2=rhs3(V+0.5*m1*dt, dx, Nx)
    n3=rhs1(V+0.5*m2*dt, dx, Nx)
    m3=rhs2(W+0.5*n2*dt, dx, Nx)
    k3=rhs3(V+0.5*m2*dt, dx, Nx)
    n4=rhs1( V+m3*dt, dx, Nx)
    m4=rhs2( W+n3*dt, dx, Nx)
    k4=rhs3( V+m3*dt, dx, Nx)
      
    W_ = W +(dt/6.)*(n1+2.*n2+2.*n3+n4)
    U_ = U +(dt/6.)*(k1+2.*k2+2.*k3+k4)
    V_ = V +(dt/6.)*(m1+2.*m2+2.*m3+m4)
    
    return U_, V_, W_


ur, xk = AdvSolve(Xc, Nx, T, c, Func, RK4)
ur2, xk2 = AdvSolve(Xc, Nx, T, c, Func, RK4)
ur3, xk3 = AdvSolve(Xc, Nx, T, c, Func, RK4)


plt.figure()
plt.plot(xk, ur[:,0])
plt.plot(xk, ur[:,25])
plt.plot(xk, ur[:,50])
show()



#figure()
#plot(tgridRK100, ur)
#plot(tgridRK200, ur2)
#plot(tgridRK400, ur3)


##########################################################
'''Analytical solution''' 
def analyt(x,t):
    return .5*np.exp(-((x-t)**2)/(2*0.1**2)) + .5*np.exp(-((x+t)**2)/(2*0.1**2))


##########################################################
an = analyt(x,0.5)
#figure()
#plot(analyt(x,0),label = 't=0')
#plot(analyt(x,0.25),label = 't=0.25')
#plot(analyt(x,0.5),label = 't=0.50')
#
#plot(ur[0],label = 't=0 for numerical')
#plot(ur[25],label = 't=0.25 for numerical')
#plot(ur[50],label = 't=0.50 for numerical')
#
#xlabel('x axis')
#ylabel('U(x)')
#title('Analytical solution compared to the RK4 method')
#legend(loc='best')
#
#figure()
#plot(analyt(x,0),label = 't=0')
#plot(analyt(x,0.25),label = 't=0.25')
#plot(analyt(x,0.5),label = 't=0.50')
#plot(U[0], label='t=0 for numerical')
#plot(U[25], label='t=0.25 for numerical')
#plot(U[50], label='t=0.50 for numerical')
#xlabel('x axis')
#ylabel('U(x)')
#title('Analytical solution compared to the Euler method')
#legend(loc='best')
#show()
#figure()   
#plot(tgridRK100, lr[:-1])
#plot(tgridRK200, (2**2)*l2r[:-1])
#
