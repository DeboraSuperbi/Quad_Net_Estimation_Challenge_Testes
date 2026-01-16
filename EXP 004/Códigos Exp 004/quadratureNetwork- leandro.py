import numpy as np #numerical computation package
import scipy as sp #library of scientific algorithms
import numba #library to compile some functions
from scipy import signal #library of scientific algorithms

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
  IFMG - v003 - Leandro Freitas (abr-2024)
  """
  # pre-allocate state vector
  x = np.empty((len(t),len(x0)), dtype=np.float64) #(len(t), len(x0))
  
  @numba.jit(nopython=True)
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
@numba.jit(nopython=True)
def quadNetOde(x, t, *args):
  """
  ODE of the Complex Network
  Inputs:
    - Adj: adjacency matrix, where the i-row is influenced by the j-column. >> Adj.shape = (N, N); where N is the number of oscillators
    - params: matrix of parameters. >> params.shape = (N, #of parameters); The parameters are: (R1, R2, C1, C2, R3, C3, R4, R5, R6, R7, Vcc, Vd, Ra, Rb, R, Rf)

  The state-space implementation is: dx = [A*x + f(x)] + [B*sum_ij]
  where A is a linear constant, f(x) a nonlinear term, and B the coupling term
  
  IFMG - v003 - Leandro Freitas (abr-2024)
  """ 
  Adj = args[0]
  params = args[1]

  x = x.reshape(-1,1)
  n = 3 #order of each oscillator
  N = len(Adj) # #of oscillators

  # call function to generate ODE function for the specific
  # dx_i/dt = A * x_i + f(x_i) + B * x(t)
  A = np.zeros((N*n, N*n), dtype=np.float64)
  Bk = np.empty((N*n, 1))
  v_max = np.empty((N, 1))
  iell1_a = np.empty_like(v_max)
  iell1_b = np.empty_like(v_max)
  iell2_a = np.empty_like(v_max)
  iell2_b = np.empty_like(v_max)
  dx=0.
  for k in np.arange(N):
          
    # obtain parameters
    R1 = params[k,0]
    R2 = params[k,1]
    C1 = params[k,2]
    C2 = params[k,3]
    R3 = params[k,4]
    C3 = params[k,5]
    R4 = params[k,6]
    R5 = params[k,7]
    R6 = params[k,8]
    R7 = params[k,9]
    Vcc = params[k,10]
    Vd = params[k,11]
    Ra = params[k,12]
    Rb = params[k,13]
    R = params[k,14]
    Rf = params[k,15]
    
    if Rf == 0.0: Ra = 99e+99

    # LINEAR TERM
    Ak = np.array( [[          0, 1/(R2*C2), 1/(R1*C1)-1/(R2*C2)+1/(Ra*C1) ],
                    [ -1/(R3*C3),         0,                             0 ], 
                    [          0, 1/(R2*C2),                  -(1/(R2*C2)) ]],
                    dtype=np.float64)
    A[k*n:(k+1)*n, k*n:(k+1)*n] = Ak # add matrix to the block diagonal structure
    
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

    # EXTERNAL INFLUENCE UPON THE CIRCUIT
    Bk[k*n:k*n+n,:] = np.array([[ Rf/(R*Ra*C1) ], # *np.sum(aij*vsin_j)
                                [ Rf/(R*Rb*C3) ], # *np.sum(aij*vcos_j)
                                [           0. ]], dtype=np.float64)


  # Matrix that define the coupling scheme, stating that vsin_j influences d(vsin_i)/dt; and vcos_j influences d(vcos_i)/dt
  couplingScheme = np.array([[1., 0., 0.],  # vsin_j ==> d(vsin_i)/dt
                             [0., 1., 0.],  # vcos_j ==> d(vcos_i)/dt
                             [0., 0., 0.]], dtype=np.float64) # no influence upon 3rd state equation
  
  Bx = np.dot(np.kron(Adj, couplingScheme), Bk*x)
  
  # NONLINEAR TERM
  # the term 'x[1::n]' takes all v_cos, of each oscillator in just one vector
  Fx = (x[1::n]<-v_max)*(x[1::n]*iell1_a + iell1_b) + (x[1::n]>v_max)*(x[1::n]*iell2_a + iell2_b)
  
  # Vector that define where the nonlinear term is located ==> in the second state equation
  nonlinearityScheme = np.array([[0],
                                 [1],  # I_ell influences the 2nd state equation
                                 [0]], dtype=np.float64)

  Fx = np.kron(Fx, nonlinearityScheme) # >> Fx.shape = (N*n, 1)

  # dx/dt = A * x + f(x) + B * x ==> terms: LINEAR Ax, NONLINEAR f(x), COUPLING Bx
  dx =  np.dot(A, x) + Fx + Bx
  return dx.reshape(-1)

####################################################################################################
# Function to implement the ODE of a complex network of quadrature oscillators in cylindrical coord
@numba.jit(nopython=True)
def quadOde_cyl(x, t, *args):
  """
  ODE of the Quadrature Circuit in Cylindrical coordinates
  Inputs:
    - params: parameters of the circuit. >> params.shape = (#of parameters); The parameters are: (R1, R2, C1, C2, R3, C3, R4, R5, R6, R7, Vcc, Vd, Ra, Rb, R, Rf)

  The state-space implementation is: dx = f(x)
  The state variables are:
    - x[0]: theta, the phase angle
    - x[1]: r, the radius
    - x[2]: v_1, the same as in the Cartesian coordinates
  
  IFMG - v000 - Leandro Freitas (abr-2024)
  """ 
  params = args[0]
  
  # obtain parameters
  R1 = params[0,0]
  R2 = params[0,1]
  C1 = params[0,2]
  C2 = params[0,3]
  R3 = params[0,4]
  C3 = params[0,5]
  R4 = params[0,6]
  R5 = params[0,7]
  R6 = params[0,8]
  R7 = params[0,9]
  Vcc = params[0,10]
  Vd = params[0,11]
  Ra = params[0,12]
  Rb = params[0,13]
  R = params[0,14]
  Rf = params[0,15]
  
  if Rf == 0.0: Ra = 99e+99

  # Nonlinear terms: v_max is used to compare with v_cos
  v_max = (R5/R4)*Vcc + ((R4+R5)/R4)*Vd
  
  # NONLINEAR TERM
  vsin = x[1]*np.sin(x[0])
  vcos = x[1]*np.cos(x[0])
  Iell = (vcos<-v_max)*((1/R5)*vcos + (1/R4)*Vcc) + (vcos>v_max)*((1/R5)*vcos - (1/R4)*Vcc)
  
  # Vector that define where the nonlinear term is located ==> in the second state equation
  dx = np.array([ [(1/(R2*C2))*np.cos(x[0])**2 +
                   (1/(R3*C3))*np.sin(x[0])**2 +
                   (1/(R1*C1*x[1])-1/(R2*C2*x[1]))*np.cos(x[0])*x[2] + 
                   (1/(C3*x[1]))*np.sin(x[0])*Iell],
                  [(1/(R2*C2)-1/(R3*C3))*x[1]*np.sin(x[0])*np.cos(x[0]) +
                   (1/(R1*C1)-1/(R2*C2))*x[2]*np.sin(x[0]) - 
                   (1/(C3))*np.cos(x[0])*Iell],  # I_ell influences the 2nd state equation
                  [(1/(R2*C2))*x[1]*np.cos(x[0])-(1/(R2*C2))*x[2]],
                  ], dtype=np.float64)

  return dx.reshape(-1)

####################################################################################################
# Function to implement the ODE of the Kuramoto approximation of the Quadrature oscillator
@numba.jit(nopython=True)
def kurNetOde(x, t, *args):
  """
  ODE of the Kuramoto approximation of the Quadrature oscillator
  Inputs:
    - Adj: adjacency matrix, where the i-row is influenced by the j-column. >> Adj.shape = (N, N); where N is the number of oscillators
    - params: matrix of parameters. >> params.shape = (N, #of parameters); The parameters are: (R1, R2, C1, C2, R3, C3, R4, R5, R6, R7, Vcc, Vd, Ra, Rb, R, Rf)

  The implementation is: dx = omega + Bx
  where omega is the natural frequencies and Bx the coupling term
  
  IFMG - v000 - Leandro Freitas (mai-2024)
  """ 
  Adj = args[0]
  params = args[1]

  x = x.reshape(-1,1)
  n = 1 #order of each oscillator
  N = len(Adj) # #of oscillators

  # call function to generate ODE function for the specific
  # dx_i/dt = A * x_i + f(x_i) + B * x(t)
  Omega = np.zeros((N*n, 1), dtype=np.float64)
  Bx = np.empty((N*n, 1))
  dx=0.
  for k in np.arange(N):
    # obtain parameters
    R1 = params[k,0]
    R2 = params[k,1]
    C1 = params[k,2]
    C2 = params[k,3]
    R3 = params[k,4]
    C3 = params[k,5]
    R4 = params[k,6]
    R5 = params[k,7]
    R6 = params[k,8]
    R7 = params[k,9]
    Vcc = params[k,10]
    Vd = params[k,11]
    Ra = params[k,12]
    Rb = params[k,13]
    R = params[k,14]
    Rf = params[k,15]
    
    if Rf == 0.0: Ra = 99e+99

    # LINEAR APPROXIMATION OF THE FREQUENCY
    freqApprox_linear = 1/(2*np.pi*np.sqrt(R1*C1*R2*C2))
    omega_k = 2*np.pi*freqApprox_linear
    
    Omega[k*n:(k+1)*n, 0] = np.array( [[ omega_k ]], dtype=np.float64)
    
    # EXTERNAL INFLUENCE UPON THE CIRCUIT
    Bx[k*n:k*n+n,:] = Rf/(R*Ra*C1)*np.dot(Adj[k*n:k*n+n,:], np.sin(x - x[k]))
  
  dx = Omega + Bx
  return dx.reshape(-1)

####################################################################################################
# Function to implement the ODE of the Kuramoto approximation of the Quadrature oscillator
@numba.jit(nopython=True)
def kurNetOde2(x, t, *args):
  """
  ODE of the Kuramoto oscillator
  Inputs:
    - Adj: adjacency matrix, where the i-row is influenced by the j-column. >> Adj.shape = (N, N); where N is the number of oscillators
    - params: vector of frequencies in Hz. >> params.shape = (N, ); The parameters are: (f1, f2, ..., fN)
    - natFreqs: vector of natural frequencies in Hz. >> natFreqs.shape = (N, ) with natFreqs = [f1, f2, ..., fN]
  
  The main difference between 'kurNetOde' and 'kurNetOde2' functions is that, in the latter, we set the natural frequencies explicitly,
  while in the former the natural frequencies are estimated based on the parameters of the circuit.

  IFMG - v000 - Leandro Freitas (mai-2024)
  """ 
  Adj = args[0]
  params = args[1]
  natFreqs = args[2]

  x = x.reshape(-1,1)
  n = 1 #order of each oscillator
  N = len(Adj) # #of oscillators

  # implement the ODE
  Omega = np.zeros((N*n, 1), dtype=np.float64)
  Bx = np.empty((N*n, 1))
  dx=0.
  for k in np.arange(N):
          
    # obtain parameters
    R1 = params[k,0]
    R2 = params[k,1]
    C1 = params[k,2]
    C2 = params[k,3]
    R3 = params[k,4]
    C3 = params[k,5]
    R4 = params[k,6]
    R5 = params[k,7]
    R6 = params[k,8]
    R7 = params[k,9]
    Vcc = params[k,10]
    Vd = params[k,11]
    Ra = params[k,12]
    Rb = params[k,13]
    R = params[k,14]
    Rf = params[k,15]
    
    if Rf == 0.0: Ra = 99e+99

    # NATURAL FREQUENCY
    omega_k = 2*np.pi*natFreqs[k]
    
    Omega[k*n:(k+1)*n, 0] = np.array( [[ omega_k ]], dtype=np.float64)
    
    # EXTERNAL INFLUENCE UPON THE CIRCUIT
    Bx[k*n:k*n+n,:] = Rf/(R*Ra*C1)*np.dot(Adj[k*n:k*n+n,:], np.sin(x - x[k]))
  
  dx = Omega + Bx
  return dx.reshape(-1)

####################################################################################################
# Compute the phase of a signal, using Hilbert transform
# @numba.jit(nopython=False)
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
# @numba.jit(nopython=False)
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
# @numba.jit(nopython=False)
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


####################################################################################################
# Compute the phase of a signal, using Hilbert transform
@numba.jit
def orderParameterPhase(phase):
  """
  Compute order parameter based on a phase variable (see https://en.wikipedia.org/wiki/Kuramoto_model)
  Inputs:
    - phase: phase variable with shape (#of points, N), being N the #of oscillators
  """
  return np.abs(np.sum(np.exp(1j*phase), axis=1)/len(phase[0,:]))
