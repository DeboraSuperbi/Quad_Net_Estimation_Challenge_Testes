import numpy as np #numerical computation package
import scipy as sp #library of scientific algorithms
from scipy import linalg
from scipy import signal
#from numba import jit #library to compile some functions

####################################################################################################
# Function to integrate ODE using Runge Kutta - intRK(odeFun, x0, t, args=()) 
def intRK(odeFun, x0, t, args=()):
    """
    Runge-Kutta integration (same parameters as scipy.integrate.odeint)
    
    Inputs:
       - odeFun: (callable) ordinary differential equation, e.g. "x, t: odeFun(x, t)"
       - x0: vector of initial states, ex. "x0 = np.array([1.0, 0.3, 3.1])" for a 3D system
       - t: np.array containing the time stamps, ex "t=np.arange(0, 100, 0.02)"
    
    Outputs:
       - x: array of states, shape (len(t), len(x0))
       - t: array of time stamps, shape (len(t))
    
    IFMG - v002 - Leandro Freitas (dez-2020)
    """
    
    # pre-allocate state vector
    x = np.empty((len(t),len(x0))) #(len(t), len(x0))
    
    #@jit(nopython=True)
   # @jit(nopython=False)
    def auxFunc(x0, t, x, args):
        # initial state
        x[0, :] = x0

        # integration step
        dt = t[1]-t[0]

        # loop to compute the states
        for k in range(1, len(t)):
            k1F = dt*odeFun(x[k-1, :], t[k], *args)
            k2F = dt*odeFun(x[k-1, :] + k1F/2., t[k], *args)
            k3F = dt*odeFun(x[k-1, :] + k2F/2., t[k], *args)
            k4F = dt*odeFun(x[k-1, :] + k3F, t[k], *args)
            # compute the actual state
            x[k, :] = x[k-1, :] + (k1F+2.*k2F+2.*k3F+k4F)/6.

        return x
    
    return auxFunc(x0, t, x, args)


####################################################################################################
# Function to implement the ODE of a complex network of quadrature oscillators
#@jit(nopython=False)
def quadNetOde(x, t, Adj, params):
  
    """
    ODE of the Complex Network
    Inputs:
    - Adj: adjacency matrix, where the i-row is influenced by the j-column. >> Adj.shape = (N, N); where N is the number of oscillators
    - params: matrix of parameters. >> params.shape = (N, #of parameters); The parameters are: (R1, R2, C1, C2, R3, C3, R4, R5, R6, R7, Vcc, Vd, Ra, Rb, R, Rf)

    The state-space implementation is: dx = [A*x + f(x)] + [B*sum_ij]
    where A is a linear constant, f(x) a nonlinear term, and B the coupling term
    """ 

    x = x.reshape(-1,1)
    n = 3 #order of each oscillator
    N = len(Adj) # #of oscillators

    # call function to generate ODE function for the specific
    # dx_i/dt = A * x_i + f(x_i) + B * x(t)
    A = np.array([], dtype=np.float64).reshape(0,0)
    Bk = np.empty((N*n,1))
    v_max = np.empty((N,1))
    iell1_a = np.empty_like(v_max)
    iell1_b = np.empty_like(v_max)
    iell2_a = np.empty_like(v_max)
    iell2_b = np.empty_like(v_max)
    for k in np.arange(N):
            
        # obtain parameters
        R1, R2, C1, C2, R3, C3, R4, R5, R6, R7, Vcc, Vd, Ra, Rb, R, Rf = params[k]
        
        if Rf == 0.0: Ra = 99e+99

        # LINEAR TERM
        Ak = np.array( [[          0, 1/(R2*C2), 1/(R1*C1)-1/(R2*C2)+1/(Ra*C1) ],
                        [ -1/(R3*C3),         0,                             0 ], 
                        [          0, 1/(R2*C2),                  -(1/(R2*C2)) ]],
                        dtype=np.float64)
        A = sp.linalg.block_diag(A, Ak) # add matrix to the block diagonal structure
        
        # NONLINEAR TERM
        # Rearrange the constants to use, vectorized, in nonlinear terms
        # Nonlinear terms: v_max is used to compare with all v_cos
        v_max[k,0] = (R5/R4)*Vcc + ((R4+R5)/R4)*Vd
        # The nonlinear terms is computed outside the for loop as
        # >> Fx = (v_cos[k]<-v_max[k])*(vcos[k]*iell1_a[k] + iell1_b[k]) + (v_cos[k]>v_max[k])*(v_cos[k]*iell2_a[k] + iell2_b[k])
        iell1_a[k,0] = -1/(C3*R6)
        iell1_b[k,0] = - Vcc/(C3*R7)
        iell2_a[k,0] = -1/(C3*R5)
        iell2_b[k,0] = + Vcc/(C3*R4)

        # EXTERNAL INFLUENCE UNPON THE CIRCUIT
        Bk[k*n:k*n+n,:] = np.array([[ (Rf/(R*Ra*C1)) ], # *np.sum(aij*vsin_j)
                                    [ (Rf/(R*Rb*C3)) ], # *np.sum(aij*vcos_j)
                                    [              0 ]],
                                    dtype=np.float64)


    # Matrix that define the coupling scheme, stating that vsin_j influences d(vsin_i)/dt; and vcos_j influences d(vcos_i)/dt
    couplingScheme = np.array([[1, 0, 0],  # vsin_j ==> d(vsin_i)/dt
                               [0, 1, 0],  # vcos_j ==> d(vcos_i)/dt
                               [0, 0, 0]]) # no influence upon 3rd state equation

    Bx = np.matmul(np.kron(Adj, couplingScheme), Bk*x)

    # NONLINEAR TERM
    # the term 'x[1::n]' takes all v_cos, of each oscillator in just one vector
    Fx = (x[1::n]<-v_max)*(x[1::n]*iell1_a + iell1_b) + (x[1::n]>v_max)*(x[1::n]*iell2_a + iell2_b)
    
    # Vector that define where the nonlinear term is located ==> in the second state equation
    nonlinearityScheme = np.array([[0],
                                   [1],  # I_ell influences the 2nd state equation
                                   [0]])

    Fx = np.kron(Fx, nonlinearityScheme) # >> Fx.shape = (N*n, 1)

    # dx/dt = A * x + f(x) + B * x ==> terms: LINEAR Ax, NONLINEAR f(x), COUPLING Bx
    dx =  np.matmul(A, x) + Fx + Bx
    return dx.reshape(-1)

####################################################################################################
# Compute the phase of a signal, using Hilbert transform
#@jit(nopython=False)
def phaseHilbert(X001, n=3):
     """
     Compute the phase variable using hilbert transform of a state vector X001. The computation is done based on the first state variable of each oscillator.
     Inputs:
       - X001: state vector with shape(#of points, N*n), being N the #of oscillators and n the #of states in each oscillator
       - n: dimension of the state vector, #of states
       """
     return np.angle(signal.hilbert(X001[:,0::n]/np.max(np.abs(X001[:,0::n]), axis=0), axis=0))


####################################################################################################
# Compute the phase of a signal, using Hilbert transform
#@jit(nopython=False)
def phase(X001, n=3):
     """
     Compute the phase variable using arctan2 of a state vector X001
     Inputs:
       - X001: state vector with shape(#of points, N*n), being N the #of oscillators and n the #of states in each oscillator
       - n: dimension of the state vector, #of states
     """
     return np.arctan2(X001[:,0::n]/np.max(np.abs(X001[:,0::n]), axis=0), X001[:,1::n]/np.max(np.abs(X001[:,1::n]), axis=0))

####################################################################################################
# Compute the phase of a signal, using Hilbert transform
#@jit(nopython=False)
def orderParameter(X001, n=3, hilbert=False):
     """
     Compute order parameter, phase coherence, for a network (see https://en.wikipedia.org/wiki/Kuramoto_model)
     Inputs:
       - X001: state vector with shape (#of points, N*n), being N the #of oscillators and n the #of states in each oscillator
       - n: dimension of the state vector, #of states
       - hilbert: use Hilbert transform to compute phase of the oscillators
     """
     if hilbert:
         phase001 = phaseHilbert(X001, n=n)
     else:
         phase001 = phase(X001, n=n)

     return np.abs(np.mean( np.exp(1j*phase001), axis=1))




