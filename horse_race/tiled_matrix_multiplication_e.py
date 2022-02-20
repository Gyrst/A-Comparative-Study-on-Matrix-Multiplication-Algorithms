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

def benchmark_tiled(f: FunType , n_list: list, s:int, N: int)->np.ndarray: #N is repetitions
    
    n_list_length = len(n_list)
    
    M: np.ndarray = np.zeros((n_list_length, N))
    # This loop takes each n in the n_list and puts in the randomly generated list
    for n in range(n_list_length):     
        A = generate_input(n_list[n])
        B = generate_input(n_list[n])
            
        if warm_up == True:
            print("warm_up")
            f(A,B,s)
            f(A,B,s)
        
        print("show_time!")
        for j in range(N):
            M[n,j] = measure(lambda: f(A,B,s))
            print("time:")
            print(M[n,j])
            if sleep: time.sleep(5)
        if sleep: time.sleep(20)
    means = np.mean(M,axis =1).reshape(n_list_length,1)
    stdevs = np.std(M,axis=1,ddof =1).reshape(n_list_length,1)
    return np.hstack ([means , stdevs ])


def run_tiled_benchmark():
    print('tiled')
    res_tiled = benchmark_tiled(tiled_multiplication, n_list, s=s, N=N)
    tiled = parent + relative_path + "tiled_multiplication_race.csv"
    write_csv(n_list, res_tiled, tiled, column_titles=column_titles_n)