import numpy as np
from scipy.stats import uniform
import scipy.sparse as s
from functions import MSE

class ESN():
  # VETTORI STATO E INPUT COLONNA
  def __init__(self, rho =0.9, Nr=100, Nu=1, r_density =0.2, i_density =0.1, Ny=1):
    #iperparametri rete
    self.rho = rho
    self.Nr = Nr
    self.Nu = Nu
    self.Ny = Ny
    self.r_density = r_density
    self.i_density = i_density

    #parametri rete
    self.W = self.build_recurrent_matrix()
    self.W_in = self.build_input_matrix()

    # maschera dropout
    self.D = np.zeros(self.W.shape)      ; self.D[:,:] = self.W[:,:] != 0 
    self.D_in = np.zeros(self.W_in.shape); self.D_in[:,:] = self.W_in[:,:] != 0 

    self.x = np.zeros((Nr,1)) # stato corrente

  def build_recurrent_matrix(self):
    wrandom = s.random(self.Nr,self.Nr,density = self.r_density, data_rvs=uniform(loc=-1,scale=2).rvs ).todense() # matrice sparsa con valori in distribuzione uniforme tra -1 e 1
    w = wrandom * ( self.rho / max(np.abs(np.linalg.eigvals(wrandom))) )
    return np.array(w)

  def build_input_matrix(self):
    w_in = s.random( self.Nr , self.Nu+1 , density = self.i_density , data_rvs=uniform(loc=-1,scale=2).rvs ).todense() # matrice sparsa con valori in distribuzione uniforme tra -1 e 1
    w_in = w_in/np.linalg.norm(w_in) # normalizzazione
    return np.array(w_in)

  def compute_state(self, u):
    u = np.vstack( (u,1) )
    z = np.dot( self.W_in, u ) + np.dot( self.W, self.x )
    output = np.tanh( z )
    self.x = output
    return np.copy( output ) # lo restituisco se serve a qualcuno da fuori

  def compute_output(self):
    return np.dot( self.Wout, np.vstack((self.x,1)) )

  def compute_output(self,u):
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

    self.Wout = np.dot( d, np.linalg.pinv(s) )

  def score(self, X, y, washout=True):
    c_out = self.compute_output
    out = np.array( list( map( c_out, X ) ) ) #shape (len(data),Ny,1)
    out = out.reshape(-1,self.Ny,1) 
    wash_len = min(int(len(X)/3),500)
    return MSE(out,y,wash_len), out 

################################# ESP INDEX #################################

def ESP_Index(esn,data,P,T):
  esn.x = np.zeros( ( esn.Nr,1 ) )
  c_state= esn.compute_state # funzione che calcola lo stato della rete
  s0 = np.array( list( map( c_state, data ) ) ) # orbita x0
  D = np.zeros( P )
  for i in range( P ):
    esn.x = np.random.rand( esn.Nr, 1 )
    si = np.array( list( map( c_state, data ) ) ) # orbita xi
    d = np.zeros( np.size(data,axis=0) - T )
    for t in range( T, np.size(data,axis=0) ):
      d[t-T] = np.linalg.norm( s0[t] - si[t] )
    D[i] = np.mean( d )
  return np.mean( D )


################################# DIMENSIONE SPAZIO STATI #################################
# calcola dimensione spazio stati 
def DSS(esn,data):
  esn.x=np.zeros((esn.Nr,1)) # resetto lo stato iniziale della rete
  c_state= esn.compute_state # funzione che calcola lo stato della rete
  l = np.array( list( map( c_state, data ) ) ) # ogni elemento della lista contiene lo stato della rete al tempo t, t=1:len(data)
  m = l.reshape(np.size(data,axis=0) , esn.Nr).T #shape(100,2000) ogni colonna corrisponde allo stato al tempo t, t=1:len(data)
  cov_m=np.cov(m) # matrice di covarianza per gli stati attraversati dalla rete: ogni unita' ricorrente vista come variabile casuale
  eigs= np.linalg.eigvalsh(cov_m)  # funzione specifica per calcolo autovalori matrici simmetriche, numericamente stabile
  dim = np.sum(eigs)**2/np.sum(np.square(eigs)) # calcolo dimensione effettiva spazio degli stati, come indicato in paper
  return dim

################################# CAPACITA DI MEMORIA #################################

# calcolo capacita' di memoria .
def MC(esn,data):
  for d in data[:1000]:
    esn.compute_state(d) # washout

  c_out = esn.compute_output # funzione che calcola output
  m = np.array( list( map(c_out,data[1000:]) ) ).reshape(1000,esn.Ny) # matrice degli yk: uno per colonna

  v1 = np.var(data[1000-2*esn.Ny:])
  MC =(np.cov(m[:,0] , data[1000:])[0,0])**2 / (v1 * np.var(m[:,0])) + sum( (np.cov( m[:,k] , data[1000-k:-k])[0,0])**2 / ( v1 * np.var(m[:,k])) for k in range(1,esn.Ny) )
  return MC


###################################################################################################
################################# ALLENAMENTO HEBBIANO ###########################################
###################################################################################################

# calcola modifiche da effettuare a matrice dei pesi w
# x: preattivazione (nel caso di Win input), y: relativa attivazione
def compute_weights(w,x,y,step):
  m = np.diag(np.square(y.reshape(-1)))
  xt = x.reshape(1,-1)
  d = step * ( np.dot(y, xt ) - np.dot(m,w) )
  return d

# esegue epoca di apprendimento hebbiano sulla matrice dei pesi di input. Mantiene densita' matrice allenata
# esn: rete da allenare, train_seq: dati per allenamento, step: learning rate
def train_input(esn,train_seq,step):
  for el in train_seq:
    preact = np.vstack((el,1))
    act = esn.compute_state(el)
    esn.W_in += np.multiply(compute_weights(esn.W_in,preact,act,step) , esn.D_in ) # applico modifiche, maschera mantiene costante la densita' 

# esegue epoca di apprendimento hebbiano sulla matrice dei pesi ricorrenti. Mantiene densita' matrice allenata
# esn: rete da allenare, train_seq: dati per allenamento, step: learning rate
def train_rec(esn,train_seq,step):
  for el in train_seq:
    preact= np.copy(esn.x)
    act = esn.compute_state(el)
    esn.W += np.multiply( compute_weights(esn.W,preact,act,step) , esn.D ) 

# esegue epoca di apprendimento hebbiano su entrambe le matrici contemporaneamente. Mantiene densita' matrici allenate
# esn: rete da allenare, train_seq: dati per allenamento, stepa: learning rates [in_step, rec_step]
def train_both(esn,train_seq,steps):
  in_step = steps[0]
  rec_step = steps[1]
  for el in train_seq:
    pin = np.vstack((el,1))
    preact= np.copy(esn.x)
    act = esn.compute_state(el)
    esn.W_in += np.multiply(compute_weights(esn.W_in,pin,act,in_step) , esn.D_in )
    esn.W += np.multiply( compute_weights(esn.W,preact,act,rec_step) , esn.D ) 

    

