from parameters import *   

### Function to measure the runtime of the implememntations
def measure(f: Callable [[],Any])->float:
    start: float = time.time()
    f()
    end: float = time.time()
    return end - start

### Find input range
def get_input_range(n):
    lower_bound = 0
    upper_bound = round(np.sqrt(2**(53)/n))
    input_range = [lower_bound, upper_bound]
    return input_range

### Generate inputs
def generate_input(n: int) -> Matrix :
    list= []
    input_range = get_input_range(n)
    for i in range(0,n*n):
        random.seed(n + i)
        l = random.randint(input_range[0],int(input_range[1]))
        list.append(float(l))
    return Matrix(n,n,np.array(list).reshape(n,n))


