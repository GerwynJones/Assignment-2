"""

Created on Mon Nov 16 13:13:43 2015

@author: c1318202

"""

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt

## RK4 Method. 
def Boundary(Ux, Nx):
    Ux[0] = Ux[Nx]
    Ux[Nx+1] = Ux[1]
    return Ux

def RK_rhs1(Vi , dx , Nx):  ## FW

    Wt = np.zeros(len(Vi))
    
    Wt[1:Nx+1] = (1./(2.*dx))*(Vi[2:Nx + 2] - Vi[0:Nx])       

    Wt = Boundary(Wt , Nx)    

    return Wt ## Returns the full array.

def RK_rhs2(Wi , dx , Nx): ## FV

    Vt = np.zeros(len(Wi))

    Vt[1:Nx+1] = (1./(2.*dx))*(Wi[2:Nx + 2] - Wi[0:Nx])       

    Vt = Boundary(Vt , Nx)      

    return Vt    


def RK_rhs3(Vi , dx , Nx): ## FU

    Ut = np.zeros(len(Vi))

    Ut[1:Nx+1] = Vi[1:Nx+1]     

    Ut = Boundary(Ut , Nx)  

    return Ut


def RK4(Ui , Wi , Vi , dx , dt , Nx):
    ## Note that boundary conditions are applied every step here.        

    l1 = RK_rhs1(Vi , dx , Nx ) #FW

    m1 = RK_rhs2(Wi , dx , Nx ) #FV   

    k1 = RK_rhs3(Vi , dx , Nx ) #FU

    
    l2 = RK_rhs1(Vi + (.5*m1*dt) , dx , Nx )

    m2 = RK_rhs2(Wi + (.5*l1*dt) , dx , Nx )

    k2 = RK_rhs3(Vi + (.5*m1*dt) , dx , Nx )


    l3 = RK_rhs1(Vi + (.5*m2*dt) , dx , Nx )

    m3 = RK_rhs2(Wi + (.5*l2*dt) , dx , Nx )    

    k3 = RK_rhs3(Vi + (.5*m2*dt) , dx , Nx )    

    
    l4 = RK_rhs1(Vi+ (m3*dt), dx , Nx )

    m4 = RK_rhs2(Wi+ (l3*dt), dx , Nx )

    k4 = RK_rhs3(Vi+ (m3*dt), dx , Nx )

   
    Wt = Wi + (dt/6.)*(l1 + (2.*l2) + (2.*l3) + l4)

    Vt = Vi + (dt/6.)*(m1 + (2.*m2) + (2.*m3) + m4)

    Ut = Ui + (dt/6.)*(k1 + (2.*k2) + (2.*k3) + k4)

    
    return Vt[1:Nx+1] , Wt[1:Nx+1] , Ut[1:Nx+1]