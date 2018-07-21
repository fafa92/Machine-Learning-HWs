from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################

  
  alpha[:,0] =pi * B[:, O[0]]
  for nn in range(1, N):
      for ss in range(S):
          alpha[ss,nn] = np.dot(alpha[:,nn-1], (A[:,ss])) * B[ss, O[nn]]
          
  return alpha


def backward(pi, A, B, O):
    
    
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  """
  
#  N = len(pi)
#  T = len(O)
#  beta = []
#
#  dT = {}
#  for i in range(1, N+1):
#      dT[i] = 0
#      beta.append(dT)
#      
#  for t in range(T - 2, -1, -1):
#      d = {}
#      o = O[t + 1]
#      
#      for i in range(1, N+1):
#          sum_seq = []
#          for j in range(1, N+1):
#              sum_seq.append(A[i][j] +B[j][o] + beta[-1][j])
#          d[i] = np.sum(sum_seq)
#      beta.append(d)
#  beta.reverse()
  S = len(pi)
  N = len(O)
  beta = np.zeros((S,N))
  
  beta[:,-1] = 1
  for nn in range((N-2),-1,-1):
      for ss in range(S):
          beta[ss,nn] = np.sum(beta[:,nn+1] * A[ss,:] * B[:, O[nn+1]])
  ###################################################
  # Q3.1 Edit here
  ###################################################
  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward algorithm

  Inputs:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
  
  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  
  
  prob=np.sum(alpha[:,-1])
  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward algorithm

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = sum((pi[y]* B[y][O[0]] * beta[y][0]) for y in range(len(pi)))

  
  ###################################################
  # Q3.2 Edit here
  ###################################################
#  N = len(pi)
#  sum_ = []
#
#  for i in range(1, N+1):
#      sum_.append(pi[i] + B[i][O[0]] + beta[0][i])
#  return np.sum(sum_)
  
  
#  prob = sum(pi[l] * B[l][O[0]] * beta[0][l] for l in O)
#  print(prob,'33333333333333444')
  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path k* (in terms of the state index)
    argmax_k P(s_k1:s_kT | x_1:x_T)
  """
  
  ###################################################
  # Q3.3 Edit here
  ###################################################
  
  path={}
  for s in range(len(pi)):
      path[s]=[]
      
      
  cur_ = {}
  for s in range(len(pi)):
      cur_[s] = pi[s]*B[s][O[0]]
  for i in range(1, len(O)):
      last_pro = cur_
      cur_ = {}
      for curr_s in range(len(pi)):
        
          mm=((last_pro[last_s]*A[last_s][curr_s]*B[curr_s][O[i]], last_s) for last_s in range(len(pi)))
          
          mmm=max(mm)
          
              
          cur_[curr_s] = mmm[0]
          path[curr_s].append(mmm[1])
          
  max_pro = -1000
  thepath = None
  for s in range(len(pi)):
      
    path[s].append(s)
    if cur_[s] > max_pro:
        

        max_pro = cur_[s]
        thepath = path[s]
    else:
        continue
  return thepath
  


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
  



def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()