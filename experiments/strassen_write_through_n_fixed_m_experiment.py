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

def benchmark_recursive(f: FunType , m_list: list, n:int, N: int)->np.ndarray: 
    
    m_list_length = len(m_list)
    
    M: np.ndarray = np.zeros((m_list_length, N))
    # This loop takes each n in the n_list and puts in the randomly generated list
    for m in range(m_list_length):
        
        A = generate_input(n)
        B = generate_input(n)
        
        if sleep: time.sleep(20)
        if warm_up:
            C = Matrix(n,n)
            f(A,B,C,m_list[m])
            C = Matrix(n,n)
            f(A,B,C,m_list[m])
        for j in range(N):
            C = Matrix(n,n)
            M[m,j] = measure(lambda: f(A,B,C,m_list[m]))
            print("time:")
            print(M[m,j])
            
            if sleep: time.sleep(5)
            
    means = np.mean(M,axis =1).reshape(m_list_length,1)
    stdevs = np.std(M,axis=1,ddof =1).reshape(m_list_length,1)
    return np.hstack ([means , stdevs ])

def benchmark_strassen(f: FunType , m_list: list, n:int, N: int)->np.ndarray:

    m_list_length = len(m_list)
    M: np.ndarray = np.zeros((m_list_length, N))
    
    # This loop takes each n in the n_list and puts in the randomly generated list
    for m in range(m_list_length):
        
        A = generate_input(n)
        B = generate_input(n)
        
        if sleep: time.sleep(20)
        if warm_up:
            f(A,B,m_list[m])
            f(A,B,m_list[m])
        
        for j in range(N):
            M[m,j] = measure(lambda: f(A,B,m_list[m]))
            print("time:")
            print(M[m,j])
            
            if sleep: time.sleep(5)
            
    means = np.mean(M,axis =1).reshape(m_list_length,1)
    stdevs = np.std(M,axis=1,ddof =1).reshape(m_list_length,1)
    return np.hstack ([means , stdevs ])


def run_write_through_fixed_n_experiment():
    # M-EXPERIMENT FOR WRITE-THROUGH
    res_write_through = benchmark_recursive(recursive_multiplication_write_through, m_list, n, N)
    title_write_through = "n" + str(n) + "_recursive_write_through_n_fixed_mtest.csv"
    relative_path_write_through = "/experiments/results/"
    full_path_write_through = parent + relative_path_write_through + title_write_through
    write_csv(m_list, res_write_through, full_path_write_through, column_titles=column_titles_m)


def run_strassen_fixed_n_experiment():
    # M-EXPERIMENT FOR STRASSEN
    res_strassen = benchmark_strassen(strassen, m_list, n, N)
    title_strassen = "n" + str(n) + "_strassen_n_fixed_mtest.csv"
    relative_path_strassen = "/experiments/results/"
    full_path_strassen = parent + relative_path_strassen + title_strassen
    write_csv(m_list, res_strassen, full_path_strassen, column_titles=column_titles_m)

