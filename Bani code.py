# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:15:08 2015

@author: c1336435
"""

from numpy import *
from matplotlib.pyplot import *
import math
close('all')
#=======================================

ti = 0.
tf = 1.2
xi = -1.0
xf = 1.
v = 2.


t = linspace(ti , tf , 122)
## This solves a 1D wave equation to see if the method works.

def rhs(U, V, W , dx, Nx , v):
    Wt = zeros(len(V))    
    Wt[1:Nx+1] = (1./(2.*dx))*(V[2:Nx+2] - V[0:Nx])  
    Wt = boundary(Wt,Nx,'periodic')
    
    Vt = zeros(len(W)) 
    Vt[1:Nx+1] = (1./(2.*dx))*(W[2:Nx+2] - W[0:Nx])  
    Vt = boundary(Vt,Nx,'periodic')
    
    Ut = zeros(len(U))
    Ut[1:Nx+1] = V[1:Nx+1] 
    Ut = boundary(Ut,Nx,'periodic')
    
    return  Ut, Vt, Wt 


    
def boundary(Ut , Nx, typebound):
    if typebound=='periodic':
        Ut[0] = Ut[Nx]
        Ut[1] = Ut[Nx+1]
    
    return Ut

def funcU(x , sigma): ## Sets initial conditions at time t = 0
    u = exp(-(x**2)/(2.*sigma**2))
    v = 0
    w = -100*x*u
    return u, v, w



def euler( Ui , Vi, Wi, dx , dt , Nx , v):
   

    temp2 = rhs(Ui, Vi, Wi, dx , Nx , v)[2]
    temp2[1:Nx+1] = Wi[1:Nx+1] + temp2[1:Nx+1]*dt    
    
    temp1 = rhs(Ui, Vi, Wi, dx , Nx , v)[1]
    temp1[1:Nx+1] = Vi[1:Nx+1] + temp1[1:Nx+1]*dt

    temp = rhs(Ui, Vi, Wi, dx , Nx , v)[0]
    temp[1:Nx+1] = Ui[1:Nx+1] + temp[1:Nx+1]*dt
    return temp, temp1, temp2





def AdvSolve(Nx , ti , tf , xi , xf , method , v ): ## v for velocity.
    
    dx = (xf - xi)/Nx
    xgrid = linspace(xi - dx , xf , Nx + 2)
    
    c = 0.5
    dt = c*dx
    Nt = int((tf - ti)/dt)    
    tgrid = linspace(ti , tf , Nt + 1)
    
    U = zeros((Nt + 1 , Nx + 2))
    V = zeros((Nt + 1 , Nx + 2))
    W = zeros((Nt + 1 , Nx + 2))
    
    ## initial conditions.
    V[0 , 1:Nx + 1] = funcU(xgrid[1:Nx+1] , 0.1)[1]
    W[0 , 1:Nx + 1] = funcU(xgrid[1:Nx+1] , 0.1)[2]
    U[0 , 1:Nx + 1] = funcU(xgrid[1:Nx+1] , 0.1)[0] ## sigma is 0.1 here.
    
    
    ## Boundary conditions
    
    U[0 , :] = boundary(U[0 , :] , Nx, 'periodic')
    V[0 , :] = boundary(V[0 , :] , Nx, 'periodic')
    W[0 , :] = boundary(W[0 , :] , Nx, 'periodic')
    
    for t in range(1 , int(Nt + 1 )):
        
        U[t , :] = method(U[t-1 , :], V[t-1 , :], W[t-1 , :], dx , dt , Nx , v)[0]           
        V[t , :] = method(V[t-1 , :], V[t-1 , :], W[t-1 , :], dx , dt , Nx , v)[1]           
        W[t , :] = method(W[t-1 , :], V[t-1 , :], W[t-1 , :], dx , dt , Nx , v)[2]           

    return U, V, W, tgrid


U, V, W, tgridE100 = AdvSolve(100, ti , tf , xi , xf , euler , v)

U2, V, W, tgridE200 = AdvSolve(200, ti , tf , xi , xf , euler , v)
U3, V, W, tgridE400 = AdvSolve(400, ti , tf , xi , xf , euler , v)

figure()
plot(U[0])
plot(U[26])
plot(U[51])

def L2norm(u1,u2,Nx1,Nt):
    norm=zeros(Nt+1)
    for i in range(Nt):
        tstep=i
        for j in range(1,Nx1):
            norm[tstep]=norm[tstep]+(u1[tstep,j]-u2[2*tstep,j*2-1])**2
    return sqrt(norm/(Nx1+1))

l = L2norm(U, U2, 100, len(tgridE100))
l2 = L2norm(U2, U3, 200, len(tgridE200))

#figure()   
#semilogy(tgridE100, l[:-1])
#semilogy(tgridE200, 2*l2[:-1])

figure()
plot(tgridE100, l[:-1])
plot(tgridE200, 2*l2[:-1])

def RK4(U, V, W, dx, dt, Nx, v):  
        
    n1=rhs(U, V,W, dx, Nx, v)[2]
    m1=rhs(U, V, W, dx, Nx, v)[1]
    k1=rhs(U, V, W, dx, Nx, v)[0]
    n2=rhs(U+(0.5)*k1*dt, V+0.5*m1*dt,W+0.5*n1*dt, dx, Nx, v)[2]
    m2=rhs(U+(0.5)*k1*dt, V+0.5*m1*dt, W+0.5*n1*dt, dx, Nx, v)[1]
    k2=rhs(U+(0.5)*k1*dt, V+0.5*m1*dt,W+0.5*n1*dt, dx, Nx, v)[0]
    n3=rhs(U+(0.5)*k2*dt, V+0.5*m2*dt,W+0.5*n2*dt, dx, Nx, v)[2]
    m3=rhs(U+(0.5)*k2*dt, V+0.5*m2*dt, W+0.5*n2*dt, dx, Nx, v)[1]
    k3=rhs(U+(0.5)*k2*dt, V+0.5*m2*dt,W+0.5*n2*dt, dx, Nx, v)[0]
    n4=rhs(U+k3*dt, V+m3*dt,W+n3*dt, dx, Nx, v)[2]
    m4=rhs(U+k3*dt, V+m3*dt, W+n3*dt, dx, Nx, v)[1]
    k4=rhs(U+k3*dt, V+m3*dt, W+n3*dt,dx, Nx, v)[0]
      
    W_ = W +(dt/6.)*(n1+2.*n2+2.*n3+n4)
    U_ = U +(dt/6.)*(k1+2.*k2+2.*k3+k4)
    V_ = V +(dt/6.)*(m1+2.*m2+2.*m3+m4)
    
    return U_, V_, W_


ur, vr, wr, tgridRK100 = AdvSolve(100, ti , tf , xi , xf , RK4 , v)
ur2, vr, wr, tgridRK200 = AdvSolve(200, ti , tf , xi , xf , RK4 , v)
ur3, vr, wr, tgridRK400 = AdvSolve(400, ti , tf , xi , xf , RK4 , v)


figure()
plot(ur[0])
plot(ur[25])
plot(ur[50])


lr = L2norm(ur, ur2, 100, len(tgridRK100))
l2r = L2norm(ur2, ur3, 200, len(tgridRK200))
#
figure()
plot(tgridRK100, lr[:-1])
plot(tgridRK200, 4*l2r[:-1])
show()

#figure()
#plot(tgridRK100, ur)
#plot(tgridRK200, ur2)
#plot(tgridRK400, ur3)

#########################################################
dx = (xf - xi)/100
c = 0.5
dt = c*dx
Nt = int((tf - ti)/dt)
x = linspace(xi - dx , xf , 100 + 2)

##########################################################
'''Analytical solution''' 
def analyt(x,t):
    return .5*exp(-((x-t)**2)/(2*0.1**2)) + .5*exp(-((x+t)**2)/(2*0.1**2))


##########################################################
an = analyt(x,0.5)
figure()
plot(analyt(x,0),label = 't=0')
plot(analyt(x,0.25),label = 't=0.25')
plot(analyt(x,0.5),label = 't=0.50')
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
