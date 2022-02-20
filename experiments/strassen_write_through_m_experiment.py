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


def benchmark_recursive(f: FunType , n_list: list, m: int, N: int)->np.ndarray: #N is repetitions
    
    n_list_length = len(n_list)
    
    M: np.ndarray = np.zeros((n_list_length, N))
    # This loop takes each n in the n_list and puts in the randomly generated list
    for n in range(n_list_length):
        
        A = generate_input(n_list[n])
        B = generate_input(n_list[n])
        if sleep: time.sleep(20)
        if warm_up:
            C = Matrix(n_list[n],n_list[n])
            f(A,B,C,m)
            C = Matrix(n_list[n],n_list[n])
            f(A,B,C,m)

        for j in range(N):
            C = Matrix(n_list[n],n_list[n])
            M[n,j] = measure(lambda: f(A,B,C,m))
            print("time:")
            print(M[n,j])
            if sleep: time.sleep(5)
    means = np.mean(M,axis =1).reshape(n_list_length,1)
    stdevs = np.std(M,axis=1,ddof =1).reshape(n_list_length,1)
    return np.hstack ([means , stdevs ])

def benchmark_strassen(f: FunType , n_list: list, m: int, N: int)->np.ndarray: #N is repetitions

    n_list_length = len(n_list)

    M: np.ndarray = np.zeros((n_list_length, N))
    # This loop takes each n in the n_list and puts in the randomly generated list
    for n in range(n_list_length):
        
        A = generate_input(n_list[n])
        B = generate_input(n_list[n])
        if sleep: time.sleep(20)

        if warm_up:
            f(A,B,m)
            f(A,B,m)

        for j in range(N):
            M[n,j] = measure(lambda: f(A,B,m))
            print("time:")
            print(M[n,j])
            if sleep: time.sleep(5)
    means = np.mean(M,axis =1).reshape(n_list_length,1)
    stdevs = np.std(M,axis=1,ddof =1).reshape(n_list_length,1)
    return np.hstack ([means , stdevs ])


def run_write_through_multiple_n_experiment():
    for m in m_list:
        res = benchmark_recursive(recursive_multiplication_write_through, n_list, m, N)
        relative_path = "/experiments/results/"
        title = parent + relative_path + str(m) + "_recursive_write_through_matrix_multiplication_mtest.csv"
        
        write_csv(n_list, res, title, column_titles=column_titles_n)


def run_strassen_multiple_n_experiment():
    for m in m_list:
        res = benchmark_strassen(strassen, n_list, m, N)
        relative_path = "/experiments/results/"
        title = parent + relative_path + str(m) + "_strassen_matrix_multiplication_mtest.csv"
        write_csv(n_list, res, title, column_titles=column_titles_n)
