# TheMatrix
**parameters.py** file contains parameters which are imported by all the other files. To add your own prefix to filename written out by running the experiments, 
change the *file_name_prefix* variable. 

To run the experiments for finding optimal s value for the tiled algorithm and the optimal m value for the write_through and strassen algorithms, 
run the file **run_experiments.py** from the main directory.

To run the horse race for all algorithms, run the file **run_horse_race.py** fro the main directory.

Directory **experiments** contains the code for tiled, strassen's and write_through experiments for finding the optimal parameters, as well as the results 
from running these experiments. It's subdirectory **plots** contains all plots.

Directory **horse_race** contains the code for horse race experiments for all algorithms, along with the results.

Directory **tests** contains the test files in jupyter notebook. 

**matrix_implementations.py** contains all algorithms implementations.

**measurement.py** contains the method to measure the runtime of the algorithms.

**plotting.ipynb** allows to create plots from all the experiment results.
