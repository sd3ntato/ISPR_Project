import pandas as pd
import plotly.graph_objects as go
import numpy as np

def MSE(out,real,wl):
  assert len(out)==len(real)
  return np.mean( np.square( out[wl:].reshape(-1) -real[wl:].reshape(-1) ) ) 

def normalize(data):
    data = data - np.mean(data)
    data = data / np.sqrt(np.var(data))
    return data

def ohlc_matrix_to_dataframe(m, n_samples=None, r=0):
  if n_samples is None:
    n_samples = np.shape(m)[0]
  d = {'Open' : m[r:n_samples+r,0],
     'High' : m[r:n_samples+r,1],
     'Low' : m[r:n_samples+r,2],
     'Close' : m[r:n_samples+r,3]
      }
  return pd.DataFrame(data = d, index = pd.RangeIndex(n_samples-r) )

def plot_candlesticks(data):
  fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                  open=data['Open'],
                  high=data['High'],
                  low=data['Low'],
                  close=data['Close'])])

  fig.show()

def plot_comparison_candlesticks_with_predicted_low(real, predicted):
  fig = go.Figure(data=[go.Candlestick(x=real.index,
                  open=real['Open'],
                  high=real['High'],
                  low=real['Low'],
                  close=real['Close'])])

  fig.add_trace(
      go.Scatter(
          x=predicted.index,
          y=predicted['Low'],
          mode="lines",
          line=go.scatter.Line(color="orange",width=1.2),
          showlegend=False)
  )
  return fig