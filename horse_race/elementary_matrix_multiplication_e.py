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


def benchmark_elementary(f: FunType , n_list: list, N: int)->np.ndarray: #N is repetitions
    
    n_list_length = len(n_list)
    
    M: np.ndarray = np.zeros((n_list_length, N))
    # This loop takes each n in the n_list and puts in the randomly generated list
    for n in range(n_list_length): 
            
        A = generate_input(n_list[n])
        B = generate_input(n_list[n])
            
        
        if warm_up == True:
            print("warm_up")
            f(A,B)
            f(A,B)
        print("Show-time!")
        for j in range(N):
            M[n,j] = measure(lambda: f(A,B))
            print("time:")
            print(M[n,j])
            if sleep: time.sleep(5)
        
        if sleep: time.sleep(20)
        
    means = np.mean(M,axis =1).reshape(n_list_length,1)
    stdevs = np.std(M,axis=1,ddof =1).reshape(n_list_length,1)
    return np.hstack ([means , stdevs])


def run_elementary_benchmark():
    print("elementary")
    res_elementary = benchmark_elementary(elementary_multiplication, n_list, N)
    elementary = parent + relative_path + "elementary_multiplication_race.csv"
    write_csv(n_list, res_elementary, elementary, column_titles=column_titles_n)
