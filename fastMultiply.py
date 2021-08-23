from numba import cuda, int8, int32, float32, int16
import numpy as np
import math
from timer import timeIt as tic
import cupy as cp

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=int16)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=int16)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float32(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sA[ty, tx] = 0
        sB[ty, tx] = 0
        if y < A.shape[0] and (tx+i*TPB) < A.shape[1]:
          sA[ty, tx] = A[y, tx + i * TPB]
        if x < B.shape[1] and (ty+i*TPB) < B.shape[0]:
          sB[ty, tx] = B[ty + i * TPB, x]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]
        # Wait until all threads finish computing
        cuda.syncthreads()
    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp

cols = 1
rows = 17
# The data array
A = np.random.randint(2, size=rows*cols, dtype = np.int16).reshape((rows, cols))
B = np.random.randint(2, size=rows*cols, dtype = np.int16).reshape((rows, cols))
A = A.T
tic('')
#Configure the memory vars
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)
C_global_mem = cuda.device_array((cols, cols),  dtype=np.int32) 
tic('memory')
# Configure the blocks
threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[1]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[0]))
blockspergrid = (blockspergrid_x, blockspergrid_y)
# Start the kernel 
fast_matmul[blockspergrid, threadsperblock](A_global_mem,B_global_mem, C_global_mem)
outFast = C_global_mem.copy_to_host()
tic('fast finder')

# Calculate with cupy
A = cp.array(A, dtype = cp.int16)
B = cp.array(B, dtype= cp.int16)
tic('')
outcp = cp.dot(A, B)
tic('cupy')

#Test if same result
outcp = cp.asnumpy(outcp)
print('equals:',np.array_equal(outFast, outcp))
print('numpy',outFast)
#print('numpy s',outFast.shape)
print('cupy',outcp)
#print('cupy s',outcp.shape)








