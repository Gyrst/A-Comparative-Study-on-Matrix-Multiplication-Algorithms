import sys
import os
  
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
sys.path.append(parent)

from parameters import *

def benchmark_tiled(f: FunType , s_list: list, n:int, N: int)->np.ndarray:

    s_list_length = len(s_list)

    M: np.ndarray = np.zeros((s_list_length, N))
    # This loop takes each n in the n_list and puts in the randomly generated list
    for s in range(s_list_length):
        
        A = generate_input(n)
        B = generate_input(n)
        
        if sleep: time.sleep(20)
        
        if warm_up:
            f(A,B,s_list[s])
            f(A,B,s_list[s])

        for j in range(N):
            M[s,j] = measure(lambda: f(A,B,s_list[s]))
            print("time:")
            print(M[s,j])
            
            if sleep: time.sleep(5)
            
    means = np.mean(M,axis =1).reshape(s_list_length,1)
    stdevs = np.std(M,axis=1,ddof =1).reshape(s_list_length,1)
    return np.hstack ([means , stdevs ])




def run_tiled_experiment():
    res_tiled = benchmark_tiled(tiled_multiplication, s_list, n, N)
    relative_path = "/experiments/results/"
    tiled = parent + relative_path + "tiled_multiplication_s_experiment.csv"
    write_csv(s_list, res_tiled, tiled, column_titles=column_titles_s)
