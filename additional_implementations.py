from matrix_implementations import *

def elementary_multiplication2(A: Matrix, B: Matrix)->Matrix:
    n = A.cols()
    C = Matrix(n,n)
    for i in range(n):
        for j in range(n):
            np.sum(A[i,:]*B[:,j])
    return C

def tiled_multiplication_fun_call(A: Matrix, B: Matrix, s: int)->Matrix:                
    n = A.cols()
    C = Matrix(n,n)

    for i in range(n//s):
        for j in range(n//s):
            for k in range(n//s):
                subA = A._arr[i*s:i*s+s,k*s:k*s+s]
                subB = B._arr[k*s:k*s+s,j*s:j*s+s]
                C._arr[i*s:i*s+s,j*s:j*s+s].__iadd__(elementary_multiplication(subA,subB))

    return C

