#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Union, Tuple, overload
import numpy as np
from numpy.lib.arraysetops import isin

class Matrix:
    """
    An implementation of a simple row-major dense matrix with 64-bit
    floating-point elements that uses Numpy as backend.
    """
    _arr: np.ndarray 
    
    _rows: int
    _cols: int
    _data: List[float]
    _first_idx: int
    _stride: int

    def __init__(self, rows: int = 0, cols: int = 0,
                     data: np.ndarray = None):
        """
        The default constructor. Specifying only rows and columns
        will create a zero-initialized matrix. If data is provided, it
        must conform to the specified rows and columns.
        """
        if data is not None:
            assert data.ndim == 2
            assert data.shape == (rows, cols)
            self._arr = data
        else:
            self._arr = np.zeros((rows,cols), dtype=np.float64)



    def rows(self)->int:
        """
        Returns the number of rows in the matrix
        """
        return self._arr.shape[0]


    
    def cols(self)->int:
        """
        Returns the number of columns in the matrix
        """
        return self._arr.shape[1]

    

    @classmethod
    def from_list(cls, data: List[List[float]])->Matrix:
        """
        Construct a matrix from a list of lists
        """
        rows = len(data)
        columns = len(data[0])
        return Matrix(rows,columns,np.array(data))



    # these overloads help mypy determine the correct types
    @overload
    def __getitem__(self, key: int)->float: ...

    @overload
    def __getitem__(self, key: Tuple[slice,slice])->Matrix: ...
        
    @overload
    def __getitem__(self, key: Tuple[int,int])->float: ...


    def __getitem__(self, key: Union[int,Tuple[int,int],slice,Tuple[int,slice], Tuple[slice,int], Tuple[slice,slice]])->Union[float,Matrix]:
        """
        Implements the operator A[i,j] supporting also slices for submatrix 
        access.

        Note however that the slice support is only partial: the step value is 
        ignored.
        """
        if isinstance(key,int):
            return self._arr.ravel()[key]
        if isinstance(key,slice):
            arr = self._arr[key]
            return Matrix(arr.shape[0], arr.shape[1], arr)
        assert isinstance(key,tuple)
        if isinstance(key[0],int) and isinstance(key[1],int):
            return self._arr[key]
        arr = self._arr[key]
        if arr.ndim == 1:
            if isinstance(key[0],int):
                arr = arr.reshape(1,-1)
            elif isinstance(key[1],int):
                arr = arr.reshape(-1,1)
        return Matrix(arr.shape[0], arr.shape[1], arr)

    
    
    def __eq__(self, that: object)->bool:
        """
        Implements the operator ==
        Returns true if and only if the two matrices agree in shape and every
        corresponding element compares equal.
        """
        if not isinstance(that, Matrix):
            return NotImplemented
        return np.array_equal(self._arr, that._arr)
    


    def __str__(self)->str:
        """
        Returns a human-readable representation of the matrix
        """
        return str(self._arr)


    def tolist(self)->List[List[float]]:
        """
        Returns a list-of-list representation of the matrix
        """
        return self._arr.tolist()
        


    def __setitem__(self, key: Union[int,Tuple[int,int]], value: float)->None:
        """
        Implements the assignment operator A[i,j] = v supporting also 
        one-dimensional flat access.

        Slices are *not* supported.
        """
        if isinstance(key,int):
            self._arr.ravel()[key] = value
        else:
            self._arr[key] = value


            
    def __add__(self, that: Matrix)->Matrix:
        
        #Regular addition of two matrices. Does not modify the operands.
        if isinstance(self, Matrix):
            m = Matrix(self.rows(), self.rows(), np.add(self._arr, that._arr))
            return m
        else: #If not a matrix but the sublevel where we just add two floats together
            return self + that

    def __iadd__(self, that: Matrix)->Matrix:
        #In-place addition of two matrices, modifies the left-hand side operand.
        if isinstance(that, Matrix):
            self._arr += that._arr
            return self
        else:
            self._arr[0] += that
            return self 


    def __sub__(self, that: Matrix)->Matrix:
        """
        Regular subtraction of two matrices. Does not modify the operands.
        """
        if isinstance(self, Matrix):
            #new matrix object that this should equal (we make a new instance of a matrix)
            m = Matrix(len(self._arr), len(self._arr), np.subtract(self._arr, that._arr))
            return m
        else: #If not a matrix but the sublevel where we just subtract two floats
            return self - that


    def __isub__(self, that: Matrix)->Matrix:
        """
        Regular subtraction of two matrices. Does not modify the operands.
        """
        self._arr -= that._arr
        return self
    


def elementary_multiplication(A: Matrix, B: Matrix)->Matrix:
    n = A.cols()
    C = Matrix(n,n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C._arr[i,j] += A._arr[i,k]*B._arr[k,j]
    return C

def transpose(A: Matrix)->None:
    a: int = A.cols()
    b = 0
    for i in range(0,a):
        for j in range(b,a):
            if(i != j):
                t = A._arr[i,j]
                A._arr[i,j] = A._arr[j,i]
                A._arr[j,i] = t
        b += 1


def elementary_multiplication_transposed(A: Matrix, B: Matrix)->Matrix:
    n = A.cols()

    C = Matrix(n,n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C._arr[i,j] += A._arr[i,k]*B._arr[j,k]

    return C


def tiled_multiplication(A: Matrix, B: Matrix, s: int)->Matrix:                
    n = A.cols()
    C = Matrix(n,n)

    for i in range(n//s):
        for j in range(n//s):
            for k in range(n//s):
                subA = A[i*s:i*s+s,k*s:k*s+s]
                subB = B[k*s:k*s+s,j*s:j*s+s]
                z = subA.cols()
                subC = Matrix(z,z)
                for l in range(z):
                    for m in range(z):
                        for o in range(z):
                            temp = subC[l,m] + subA[l,o]*subB[o,m]
                            subC[l,m] = temp
                C[i*s:i*s+s,j*s:j*s+s].__iadd__(subC)

    return C


def recursive_multiplication_copying(A:Matrix , B:Matrix) -> Matrix:
            
    n = A.rows()
    
    if A.rows() == 1:
    
        C = A[0]*B[0]
        return C
    
    else:
        C = Matrix(A.rows(), A.cols())
        
        C00 = C[:n//2,:n//2]
        C01 = C[:n//2,n//2:]
        C10 = C[n//2:,:n//2]
        C11 = C[n//2:,n//2:]


        P0 = A[:n//2,:n//2]
        P1 = A[:n//2,n//2:]
        P2 = A[:n//2,:n//2]
        P3 = A[:n//2,n//2:]
        P4 = A[n//2:,:n//2]
        P5 = A[n//2:,n//2:]
        P6 = A[n//2:,:n//2]
        P7 = A[n//2:,n//2:]
        
        Q0 = B[:n//2,:n//2]
        Q1 = B[n//2:,:n//2]
        Q2 = B[:n//2,n//2:]
        Q3 = B[n//2:,n//2:]
        Q4 = B[:n//2,:n//2]
        Q5 = B[n//2:,:n//2]
        Q6 = B[:n//2,n//2:]
        Q7 = B[n//2:,n//2:]

        M0 = recursive_multiplication_copying(P0, Q0)
        M1 = recursive_multiplication_copying(P1, Q1)
        M2 = recursive_multiplication_copying(P2, Q2)
        M3 = recursive_multiplication_copying(P3, Q3)
        M4 = recursive_multiplication_copying(P4, Q4)
        M5 = recursive_multiplication_copying(P5, Q5)
        M6 = recursive_multiplication_copying(P6, Q6)
        M7 = recursive_multiplication_copying(P7, Q7)

        C00 += M0 + M1
        C01 += M2 + M3
        C10 += M4 + M5
        C11 += M6 + M7

        return C


def elementary_multiplication_in_place(A: Matrix, B: Matrix, C: Matrix)-> Matrix:
    
    #Description

    """
    An auxiliary function that computes elementary matrix
    multiplication in place, that is, the operation is C += AB such
    that the product of AB is added to matrix C.
    """

    n = A.cols()

    #This is the optimized algorithm using the i, k, j structure --> however the gain described
    # and seen when we run it in Java, we do not get in python.
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C._arr[i,j] += A._arr[i,k] * B._arr[k,j]
    return C

def recursive_multiplication_write_through(A: Matrix, B: Matrix, C:Matrix, m=0)->Matrix:
    
    #Instructions:
        # Computes C=AB recursively using a write-through strategy. That
        # is, no intermediate copies are created; the matrix C is
        # initialized as the function is first called, and all updates
        # are done in-place in the recursive calls.
        
        # The parameter m controls such that when the subproblem size
        # satisfies n <= m, * an iterative cubic algorithm is called instead.

    #initializing C and getting the length of n
    n = A.rows()
        
    if n <= m:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C._arr[i,j] += A._arr[i,k] * B._arr[k,j]
        return C

    elif n == 1:
        C._arr[0] += A._arr[0]*B._arr[0]
        return C
    
    else:
        #M0 C upper left                                     a00             b00             c00
        a00b00 = recursive_multiplication_write_through(A[:n//2,:n//2], B[:n//2,:n//2], C[:n//2,:n//2], m)
        #M1 C upper left                                     a01             b10             c00
        a01b10 = recursive_multiplication_write_through(A[:n//2,n//2:], B[n//2:,:n//2], C[:n//2,:n//2], m)
        
        #M2 C upper right                                    a00             b01             c01
        a00b01 = recursive_multiplication_write_through(A[:n//2,:n//2], B[:n//2,n//2:], C[:n//2,n//2:], m)
        #M3 C upper right                                    a01             b11             c01
        a01b11 = recursive_multiplication_write_through(A[:n//2,n//2:], B[n//2:,n//2:], C[:n//2,n//2:], m)
        
        #M4 C lower left                                     a10             b00             c10
        a10b00 = recursive_multiplication_write_through(A[n//2:,:n//2], B[:n//2,:n//2], C[n//2:,:n//2], m)
        #M5 C lower left                                     a11             b10             c10
        a11b10 = recursive_multiplication_write_through(A[n//2:,n//2:], B[n//2:,:n//2], C[n//2:,:n//2], m)
        
        #M6 C lower right                                    a10             b01             c11
        a10b01 = recursive_multiplication_write_through(A[n//2:,:n//2], B[:n//2,n//2:], C[n//2:,n//2:], m)
        #M7 C lower right                                    a11             b11             c11
        a11b11 = recursive_multiplication_write_through(A[n//2:,n//2:], B[n//2:,n//2:], C[n//2:,n//2:], m)

        return C



def strassen(A: Matrix, B: Matrix, m=0)->Matrix:
    """
    Computes C=AB using Strassen's algorithm. The structure ought
    to be similar to the copying recursive algorithm. The parameter
    m controls when the routine falls back to a cubic algorithm, as
    the subproblem size satisfies n <= m.
    """
    n = A.rows()
    
    if n <= m:
        C = Matrix(n,n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C._arr[i,j] += A._arr[i,k]*B._arr[k,j]
        return C    
    elif n == 1:
        C = A._arr[0]*B._arr[0]
        return C
    
    else:
        #Making a C responding to the current size of A in the current recursion level
        C = Matrix(A.rows(), A.rows())
        
        #Cutting our copy of C into four sections
        C00 = C[:n//2,:n//2]
        C01 = C[:n//2,n//2:]
        C10 = C[n//2:,:n//2]
        C11 = C[n//2:,n//2:]
        
        
        # P1 = A00 + A11
        P1 = A[:n//2,:n//2] + A[n//2:,n//2:]
        # P2 = A10 + A11
        P2 = A[n//2:,:n//2] + A[n//2:,n//2:]
        # P3 = A00
        P3 = A[:n//2,:n//2]
        # P4 = A11
        P4 = A[n//2:,n//2:]
        # P5 = A00 + A01
        P5 = A[:n//2,:n//2] + A[:n//2,n//2:]
        # P6 = A10 - A00
        P6 = A[n//2:,:n//2] - A[:n//2,:n//2]
        # P7 = A01 - A11
        P7 = A[:n//2,n//2:] - A[n//2:,n//2:]
        
        # Q1 = B00 + B11
        Q1 = B[:n//2,:n//2] + B[n//2:,n//2:]
        # Q2 = B00
        Q2 = B[:n//2,:n//2]
        # Q3 = B01 - B11 
        Q3 = B[:n//2,n//2:] - B[n//2:,n//2:]
        # Q4 = B10 - B00
        Q4 = B[n//2:,:n//2] - B[:n//2,:n//2]
        # Q5 = B11
        Q5 = B[n//2:,n//2:]
        # Q6 = B00 + B01
        Q6 = B[:n//2,:n//2] + B[:n//2,n//2:]
        # Q7 = B10 + B11
        Q7 = B[n//2:,:n//2] + B[n//2:,n//2:]
            
        # Then compute Mi = Pi*Qi by a recursive application of the function
        M1 = strassen(P1,Q1, m)
        M2 = strassen(P2,Q2, m)
        M3 = strassen(P3,Q3, m)
        M4 = strassen(P4,Q4, m)
        M5 = strassen(P5,Q5, m)
        M6 = strassen(P6,Q6, m)
        M7 = strassen(P7,Q7, m)
        
        # Following the recipe from the slides:
        
        C00 += M1 + M4 - M5 + M7
        C01 += M3 + M5
        C10 += M2 + M4
        C11 += M1 - M2 + M3 + M6

        return C


