import numpy as np
from scipy.stats import uniform
import scipy.sparse as s
from utils import MSE

class ESN():
  def __init__(self, rho =0.9, Nr=100, Nu=1, r_density =0.2, i_density =0.1, Ny=1):
    #hyperparameters
    self.rho = rho # spectral radius of recurrent matrix
    self.Nr = Nr # recurrent units
    self.Nu = Nu # input units
    self.Ny = Ny # output units
    self.r_density = r_density # density of recurrent matrix
    self.i_density = i_density # density of input matrix

    #parametri rete
    self.W = self.build_recurrent_matrix() # recurrent matrix
    self.W_in = self.build_input_matrix() # inupt matrix

    self.x = np.zeros((Nr,1)) # current state

  def build_recurrent_matrix(self):
    # create sparse matrix with values in [-1,1]
    wrandom = s.random(self.Nr,self.Nr,density = self.r_density, data_rvs=uniform(loc=-1,scale=2).rvs ).todense()
    # rescale it to set spectral radius
    w = wrandom * ( self.rho / max(np.abs(np.linalg.eigvals(wrandom))) )
    return np.array(w)

  def build_input_matrix(self):
    # create sparse matrix with values in [-1,1]
    w_in = s.random( self.Nr , self.Nu+1 , density = self.i_density , data_rvs=uniform(loc=-1,scale=2).rvs ).todense()
    w_in = w_in/np.linalg.norm(w_in) # normalizzazione
    return np.array(w_in)

  def compute_state(self, u):
    # implementation of state update equations
    u = np.vstack( (u,1) ) # to use bias i add 1 as constant activation of bias unit
    z = np.dot( self.W_in, u ) + np.dot( self.W, self.x )
    output = np.tanh( z )
    self.x = output
    return np.copy( output ) # return state in case i need it outside

  def compute_output(self, u=None):
    if u is None:
      return np.dot( self.Wout, np.vstack((self.x,1)) )
    else:
      return np.dot( self.Wout, np.vstack((self.compute_state(u),1)) )

  def train(self,train_x,train_y,wash_seq):
    for d in wash_seq:
      self.compute_state(d) # washout

    # collect states
    c_state = self.compute_state
    s = np.array( list( map( c_state, train_x ) ) ) # shape(len(data),Nr,1)
    s = np.hstack( [ np.vstack((t,1)) for t in s] )
    assert np.shape(s) == (self.Nr+1, np.size(train_x,axis=0)), f'{np.shape(s)}'

    # collect outputs
    d = np.hstack( [t for t in train_y] )
    assert np.shape(d) == (self.Ny,np.size(train_y,axis=0))

    # compute output matrix
    self.Wout = np.dot( d, np.linalg.pinv(s) )

  # compute mean squared error and outputs
  def score(self, X, y, washout=True):
    c_out = self.compute_output
    out = np.array( list( map( c_out, X ) ) ) #shape (len(data),Ny,1)
    out = out.reshape(-1,self.Ny,1) 
    wash_len = min(int(len(X)/3),500)
    return MSE(out,y,wash_len), out 

