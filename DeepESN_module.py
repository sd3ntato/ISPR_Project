import numpy as np
from scipy.stats import uniform
import scipy.sparse as s
from utils import MSE

class Reservoir():
  def __init__(self, rho=0.9 , Nu=10, Nr=10,r_density=0.5, i_density=1):
    self.rho = rho # spectral radius of recurrent matrix
    self.Nr = Nr # number of recurrent units
    self.Nu = Nu # number of input units
    self.r_density = r_density # density of recurrent matrix
    self.i_density = i_density # density of input matrix

    self.W = self.build_recurrent_matrix() # recurrent matrix
    self.W_in = self.build_input_matrix()  # input matrix
    
    self.x = np.zeros((Nr,1)) # current state
  
  def build_recurrent_matrix(self):
    # create sparse matrix with values in [-1,1]
    wrandom = s.random(self.Nr,self.Nr,density = self.r_density, data_rvs=uniform(loc=-1,scale=2).rvs ).todense() # matrice sparsa con valori in distribuzione uniforme tra -1 e 1
    # rescale it to set spectral radius
    w = wrandom * ( self.rho / max(np.abs(np.linalg.eigvals(wrandom))) )
    return np.array(w)

  def build_input_matrix(self):
    # create sparse matrix with values in [-1,1]
    w_in = s.random( self.Nr , self.Nu+1 , density = self.i_density , data_rvs=uniform(loc=-1,scale=2).rvs ).todense() # matrice sparsa con valori in distribuzione uniforme tra -1 e 1
    w_in = w_in/np.linalg.norm(w_in) # normalizzazione
    return np.array(w_in)

  def compute_state(self, u):
    u = np.vstack((u,1))
    z = np.dot( self.W_in, u ) + np.dot( self.W, self.x )
    output = np.tanh(z)
    self.x = output
    return np.copy(output) # lo restituisco se serve a qualcuno da fuori
    

class DeepESN():
  def __init__(self, rho =0.9, N=10, Nr=10, Nu=1, Ny=1, r_density=0.5, i_density=1):
    #iperparametri rete
    self.rho = rho # spectral radius of recurrent matrices
    self.N = N # number of reservoirs
    self.Nu = Nu # number of input units
    self.Nr = Nr # number of recurrent units
    self.Ny = Ny # number of output units

    # build the stack of reservoirs
    self.ress = [None]*N
    self.ress[0] = Reservoir(Nu=Nu, Nr=Nr, r_density=r_density, i_density=i_density, rho=rho)
    for i in range(1,N):
      self.ress[i] = Reservoir(Nu=Nr,Nr=Nr, r_density=r_density, i_density=i_density, rho=rho) 
    
  def compute_state(self,u):
    cu = self.ress[0].compute_state(u) 
    for i in range(1,self.N):
      cu = self.ress[i].compute_state(cu)
    return np.array( list( np.copy(res.x) for res in self.ress ) ) # shape (N,Nr,1)
  
  # compute output of the concat-readout
  def compute_output(self, u=None):
    if u is None:
      x_c = np.vstack( list( self.ress[i].x for i in range( self.N) ))
      return np.dot( self.Wout, np.vstack((x_c,1)) )
    else:
      x_c = self.compute_state(u).reshape(-1,1)
      return np.dot( self.Wout, np.vstack((x_c,1)) )
  
  # compute output of the readout attached to the i-th reservoir in the stack
  def compute_output_i(self,u,i):
  	x = self.compute_state(u)[i]
  	return np.dot( self.ress[i].Wout, np.vstack((x,1)) )
  
  # train a readout that is attached to all recurrent units (all reservoirs)
  def train_concat(self,train_x,train_y,wash_seq):
    for d in wash_seq:
      self.compute_state(d) # washout

    # collect states
    c_state = self.compute_state
    s = np.array( list( map( c_state, train_x ) ) ) #shape ( len(data) , esn.N , esn.Nr , 1 )
    s = np.hstack( [ np.vstack((t.reshape(-1,1),1)) for t in s] )
    assert np.shape(s) == ((self.Nr*self.N)+1, np.size(train_x,axis=0)), f'{np.shape(s)}'
    
    # collect outputs
    d = np.hstack( [t for t in train_y] )

    self.Wout = np.dot( d, np.linalg.pinv(s) )

  # train i-th readout
  def train(self,train_x,train_y,wash_seq,i): # allena il readout dell'i-esimo reservoir
    for d in wash_seq:
      self.compute_state(d) # washout

    # collect states
    s = np.array( list( self.compute_state(d)[i] for d in train_x ) ) #shape ( len(data) , esn.N , esn.Nr , 1 )
    s = np.hstack( [ np.vstack((t,1)) for t in s] )

    # collect outputs
    d = np.hstack( [t for t in train_y] )

    self.ress[i].Wout = np.dot( d, np.linalg.pinv(s) )
      
  def score_concat(self, X, y, washout=True):
    c_out = self.compute_output
    out = np.array( list( map( c_out, X ) ) ) #shape (len(data),Ny,1)
    out = out.reshape(-1,self.Ny,1) 
    wash_len = min(int(len(X)/3),500)
    return MSE(out,y,wash_len), out 
  
  def score(self, X, y, i, washout=True):
    out = np.array( list( [ self.compute_output_i( x, i ) for x in X ] ) ) #shape (len(data),Ny,1)
    out = out.reshape(-1,self.Ny,1) 
    wash_len = min(int(len(X)/3),500)
    return MSE(out,y,wash_len), out 

  def reset_states(self):
    for res in self.ress:
      res.x = np.zeros((self.Nr,1))
