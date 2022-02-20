import csv
import numpy as np
import sys
import time
import random
from typing import List , Tuple , Optional , Dict , Callable , Any
import copy
from matrix_implementations import *
from measurement import measure, generate_input

### For benchmark functions
OptTuple3i = Optional[Tuple[int ,int ,int]]
FunType = Callable [[List[int]], OptTuple3i]

### Parameters enabling runing on different computers
sleep = False
warm_up = True
file_name_prefix = "GG" # enter your prefix
relative_path = "/horse_race/results/"



## parameters for all the experiment. Modify here to run experiments with different parameters.
n_list = [2,4,8,16,32,64,128,256]  ### n list for experiment with multiple values of m and n for strassen & write_through
m_list = [0,2,4,8,16,32,64,128]    ### m list to find the optimal m parameter for n=256 for strassen & write_through
s_list = [2,4,8,16,32,64,128]      ### s list for tiled experiment - finding optimal value of s
n = 256
N = 3
s = 32 
m_strassen = 8
m_write_trhough = 16


column_titles_s = ['s', 'time(s)', 'stdv']
column_titles_m = ['m', 'time(s)', 'stdv']
column_titles_n = ['n', 'time(s)', 'stdv']

def write_csv(n_list: list, res: np.ndarray, filename: str, column_titles:str=None):
    """write_csv

    Args:
        n_list (list): list of n (the matrix side length) that the the experiment is run with
        res (np.ndarray): results from the experiment
        filename (str): the filename that you desire
        column_titles (lst): takes a list with the columns title for the csv file. The titles should be given comma seperated words and no spaces
    """
    with open(filename ,'w') as f:
        writer = csv.writer(f)
        if column_titles != None:
            writer.writerow(column_titles)
        for i in range(len(n_list)):
            writer.writerow ([n_list[i]] + res[i,:].tolist())