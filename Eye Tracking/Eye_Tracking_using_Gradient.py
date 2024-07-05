import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2 as cv
import numba
from numba import cuda, njit, prange
import math
import time

# Part - 1

# Load the iris image (replace 'iris_image.jpg' with your image file)
#iris_image = cv.imread('eye_image_sample.jpg', 0)
iris_image = cv.imread('Eye_closeup_720p.jpg', 0)    # Eye_closeup_720p.jpg
iris_image = cv.convertScaleAbs(iris_image, alpha=3, beta=25)   # alpha - Scaling Factor (Contrast),   beta - Offset (Brightness)(-25 to make pupil absolute intensity)

# Thresholding:
#(T, iris_image) = cv.threshold(iris_image, 45, 255, cv.THRESH_BINARY)

# Remove Noise
iris_image_smooth = cv.GaussianBlur(iris_image, (7,7), 0)


cv.imshow("image", iris_image)  # Original Image
cv.imshow("image_blurred_gaussian", iris_image_smooth)   # Smooth Image
#cv.createTrackbar('Brightness', 'image', 255, 2 * 255)  
cv.waitKey(0) 
cv.destroyAllWindows()

print(f'shape of image: {np.shape(iris_image)}')

# Define parameters    
R_max = 85  # Maximum radius  #85           # radius is around 76
R_min = 70  # Minimum radius  # 70
beta = 2  # Weighting factor for horizontal gradients
lambda_value = 0.95  # Weight for combining gradient and intensity



# Initialize correlation output (CO) as zeros
CO = np.zeros_like(iris_image_smooth, dtype=np.float32)


# Step 1: Create O_coa (Complex Operator)
O_coa = np.zeros((R_max*2+3, R_max*2+3), dtype=np.complex64)     # Radius = R_max+1
for m in range(-R_max, R_max + 1): 
    for n in range(-R_max, R_max + 1):
        #print(f'{m},{n} | {m + R_max}, {n + R_max}')
        radius = np.sqrt(m ** 2 + n ** 2)
        if (radius >= R_min and radius <= R_max):
            theta = np.arctan2(-m, n)                # n: x-axis | m: y-axis
            O_coa[m + R_max+1, n + R_max+1] = (1/((2*R_max+3)**2)) * (np.cos(theta) + 1j * np.sin(theta)) / radius   # Divide by (2*R_max+3)**2  so overall convolution sum is divided by 2*R_max+3)**2

print(f'shape of O_coa = ',np.shape(O_coa))
print(f'value of O_coa: ', (O_coa ))

np.savetxt('kernels_O_coa.txt', O_coa)    #



# # Phase Coded Analus Kernel:
# P_ca = np.zeros((R_max*2+3, R_max*2+3), dtype=np.complex64)     # Radius = R_max+1
# for m in range(-R_max, R_max + 1): 
#     for n in range(-R_max, R_max + 1):
#         #print(f'{m},{n} | {m + R_max}, {n + R_max}')
#         radius = np.sqrt(m ** 2 + n ** 2)
#         if (radius >= R_min and radius <= R_max):
#             theta =  2*math.pi * (math.log(radius)-math.log(R_min))/(math.log(R_max)-math.log(R_min))               
#             P_ca[m + R_max+1, n + R_max+1] = R_max * (np.cos(theta) + 1j * np.sin(theta)) / radius


# K_oa_pca_x = np.real(O_coa) * P_ca
# np.savetxt('kernels_K_oa_pca.txt', K_oa_pca_x) 
# K_oa_pca_y = np.imag(O_coa) * P_ca

# print(K_oa_pca_x)



# Create a Schaar kernel (3x3)
S_x = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.float32)
S_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.float32)



# I_x = cv.filter2D(iris_image_smooth, -1, S_x)
# I_y = cv.filter2D(iris_image_smooth, -1, S_y)

# output after Phase Coded Annulus:
# output = cv.filter2D(I_x, -1, abs(K_oa_pca_x)) + cv.filter2D(I_y, -1, abs(K_oa_pca_y))


# Step 2: Create the Gradient Complex Orientation Annulus (G_gcoa): Perform convolution with real and imaginary parts separately
G_gcoa = beta * cv.filter2D(np.real(O_coa), -1, S_x) + 1/beta * cv.filter2D(np.imag(O_coa), -1, S_y)

print(type(G_gcoa))
print(f'value of G_coa: ', (G_gcoa ))

np.savetxt('kernels_G_gcoa.txt', G_gcoa)   #


# Step 3: Create Weighting Matrix Kernel (W_a)
W_a = np.zeros((R_max*2+3, R_max*2+3), dtype=np.float32)      #Radius R_max+2
for m in range(-R_max, R_max + 1):
    for n in range(-R_max, R_max + 1):
        radius = np.sqrt(m ** 2 + n ** 2)
        if radius <= R_max:
            if ((m == 0) and (n == 0)):
                W_a[m + R_max, n + R_max] = (1/((2*R_max+3)**2))  # Divide by (2*R_max+3)**2  so overall convolution sum is divided by 2*R_max+3)**2
                
            else:
                W_a[m + R_max+1, n + R_max+1] = (1/((2*R_max+3)**2)) / np.sqrt(m ** 2 + n ** 2)   # Divide by (2*R_max+3)**2  so overall convolution sum is divided by 2*R_max+3)**2

print(f'value of W_a: ', (W_a ))

np.savetxt('kernels_W_a.txt', W_a)    #

np.savetxt(f'kernels_255-image.txt', 255 - iris_image_smooth)
print(f'SIZE 255-Image: {np.shape(iris_image_smooth)}')

# Step 4: Compute Average Intensity
W =  cv.filter2D((255 - iris_image_smooth), -1, W_a)
print(f'shape of W: {np.shape(W)}')
print(f'value of W: ', (W ))

np.savetxt('kernels_W.txt', W)   #


## Trial:
CO_image_gcoa = cv.filter2D(iris_image_smooth, -1, G_gcoa)    # Delete this later: this is just to check the output

# Step 5: Combine Gradient and Intensity for Correlation Output (CO)
CO = lambda_value * cv.filter2D(iris_image_smooth, -1, G_gcoa) + (1 - lambda_value) * W
print(f'shape of CO: {np.shape(CO)}')

np.savetxt('kernels_CO.txt', CO)    #




fig1 = plt.figure()
ax1 = fig1.add_subplot(321)
ax2 = fig1.add_subplot(322)
ax3 = fig1.add_subplot(323)
ax4 = fig1.add_subplot(324)
ax5 = fig1.add_subplot(325)
ax6 = fig1.add_subplot(326)


# 'bilinear' interpolation - this will look blurry
# 'nearest' interpolation - faithful but blocky
ax1.imshow(iris_image, cmap=cm.Greys_r)
ax1.set_title('Input Eye Image')
ax2.imshow(G_gcoa, interpolation='nearest', cmap=cm.Greys_r)
ax2.set_title('G_gcoa')
ax3.imshow(W_a, interpolation='nearest', cmap=cm.Greys_r) 
ax3.set_title('W_a')
ax4.imshow(W, interpolation='nearest', cmap=cm.Greys_r) 
ax4.set_title('W')
ax5.imshow(CO_image_gcoa) 
ax5.set_title('CO_image_gcoa (No W)')
ax6.imshow(CO) 
ax6.set_title('CO: Final Output')

# Showing the CO output np array as an image:
#plt.imshow(CO)
plt.show()




# ----------------------------------------------------------------------------------------------------------------------------

# Part - 2
# Find Local Maximas: using Numba CUDA (Fast Result: Parallel Processing)

@cuda.jit
def find_maxima(inp, maxima_array, threshold, index):
    """
    inp:  Input 2D Array (Image)
    maxima_array:  2D array which stores coordinates of the Maxima values above the 'threshold' in the image
    index:  1D array, act as index of the 'maxima_array'. Taken as 1D array instead of a int value so the 'cuda.atomic.add(index, 0, 1)' can be used to sequentially update this index value
    """
    ix, iy = cuda.grid(2)   # The first index is the fastest dimension: Gives unique value correspond to each thread
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)   # threads per grid dimension = Block Dimension * Grid Dimension

    n0, n1 = inp.shape      # The last index is the fastest dimension

    i = 0
    # Stride each dimension independently
    for i0 in range(iy, n0, threads_per_grid_y):
        for i1 in range(ix, n1, threads_per_grid_x):
            if (inp[i0, i1] >= threshold):
                #maxima_array[i] = (i0, i1)
                # tmp = maxima_array[i]
                # i = i+1
                current_index = cuda.atomic.add(index, 0, 1)   # "Sequentially" increase the index value by 1 for all threads so they don't increase this together at same time
                maxima_array[current_index] = (i0, i1)         # This will run only when 'current_index' value gets updated i.e will wait until 'current_index' value gets updated
                                                               # (sequentially: so multiple threads won't overwrite/access same index together)
    
    


size_y, size_x = np.shape(CO)

threads_per_block_2d = (16, 16)    # 256 threads total per block
blocks_per_grid_2d = ((size_x + (threads_per_block_2d[0] - 1)) // threads_per_block_2d[0], (size_y + (threads_per_block_2d[1] - 1)) // threads_per_block_2d[1])      
print('blocks_per_grid_2d: ', blocks_per_grid_2d)


CO_gpu = cuda.to_device(CO)     # Copy variable to GPU

number_of_maxima = 300      # Number of Maxima we want to consider
maxima_array = np.zeros((number_of_maxima, 2), dtype=np.int32)
CO_maxima_array_gpu = cuda.to_device(maxima_array)   # Copy variable to GPU


threshold = np.float32(0.8)        # Threshold above which will be called Maxima
# threshold_gpu = cuda.to_device(threshold)


index = cuda.to_device(np.array([0], dtype=np.int32))    # 1D array, act as index of the 'maxima_array'. Taken as 1D array instead of a int value so the 
                                                         # 'cuda.atomic.add(index, 0, 1)' can be used to sequentially update this index value



find_maxima[blocks_per_grid_2d, threads_per_block_2d](CO_gpu, CO_maxima_array_gpu, threshold, index)


CO_maxima_array = CO_maxima_array_gpu.copy_to_host()    # Copy to CPU  : Maxima Locations Array

#print('CO_maxima_array: ', CO_maxima_array)




# Time Performance Measurement:
"""When using Numba, we have one detail we must pay attention to. Numba is a Just-In-Time compiler, meaning that the functions are only compiled when they are called. 
Therefore timing the first call of the function will also time the compilation step which is in general much slower. We must remember to always compile the code first 
by launching the kernel and then synchronizing it to ensure that nothing is left to run in the GPU. This ensures that the next kernel runs immediately without compilation."""


# from time import perf_counter_ns

# # Compile and then clear GPU from tasks
# find_maxima[blocks_per_grid_2d, threads_per_block_2d](CO_gpu, CO_maxima_array_gpu, threshold, index)
# cuda.synchronize()

# timing = np.empty(31)
# for i in range(timing.size):
#     tic = perf_counter_ns()  # returns in 'ns' (Nano-seconds)
#     find_maxima[blocks_per_grid_2d, threads_per_block_2d](CO_gpu, CO_maxima_array_gpu, threshold, index)
#     cuda.synchronize()
#     toc = perf_counter_ns()
#     timing[i] = toc - tic
# timing *= 1e-3  # convert to μs

# print(f"Elapsed time: {timing.mean():.0f} ± {timing.std():.0f} μs")

" Elapsed time: 128 ± 6 μs "



# # CPU Time:

# def find_maxima_cpu(inp, maxima_array, threshold):
#     n0, n1 = inp.shape
#     i = 0
#     for i in range (0, n0, 1):
#         for j in range (0, n1, 1):
#             if (inp[i, j] >= threshold):
#                 maxima_array[i] = (i, j)
#                 i += 1
#     return maxima_array

# maxima_array_cpu = np.zeros((300, 2), dtype=np.int32)

# start_time_cpu = time.time()   # gives in 'seconds'
# arr = find_maxima_cpu(CO, maxima_array_cpu, 0.8)
# end_time_cpu = time.time()
    
# print("CPU Time: ", (end_time_cpu-start_time_cpu) * 10**3, "ms")

" CPU Time: 39 ms"

" 300 times speed increase !! (by CUDA GPU)"



# Estimating Mean & SD around the Maxima points within (11 x 11) window:
""" Allotting 300 Blocks for the 300 Maxima locations and each Block has 12x12 Threads (1 block for each maxima location around which we are going to
    estimate 'Mean' and 'SD' in 11x11 window:  so 1 thread for each element in the window) """
threads_per_block_2d = (16, 16)    # 256 threads        # don't do 12,12 because not all values are getting added in the parallel sum technique used
blocks_per_grid = 300     # = Number of Considered Maximas 

@cuda.jit
def maxima_mean_std(inp, maxima_array, kernel_size, psr):
    
    bid = cuda.blockIdx.x       # Block ID
    tid_x = cuda.threadIdx.x       # Thread ID X
    tid_y = cuda.threadIdx.y       # Thread ID Y

    k_center_y, k_center_x = maxima_array[bid]      # Each Block ID corresponds to a maxima location coordinate (which is going to be the KERNEL CENTER for the 11x11 Window)

    if (tid_x < kernel_size and tid_y < kernel_size):  # Fill value of the image location corresponding to each of the window location around the maxima location as center (upto 0-10 thread, rest threads (16-11= 5) along x & y remains vacant (0))
        s_thread = inp[k_center_y - (int(kernel_size/2) - tid_y), k_center_x - (int(kernel_size/2) - tid_x)]
    else:
        s_thread = 0      # rest threads (16-11= 5) along x & y remains vacant (0) since threads are 16 but window size is 11: (256-144 = 112 threads vacant)


    # Allocate shared array
    """ Threads are independent and cannot access each other's values. To sum all the values in all the threads, we need to create a
        'Shared Array' and put the thread values into this shared array so all its value's sum can be calculated. """
    s_block = cuda.shared.array((threads_per_block_2d[1], threads_per_block_2d[0], 2), numba.float32)   # 3D Shared Array: ** Have to give CONST (not any parameter, so here global constant given)
    """  3D Shared Array created: TWO 2-D Arrays: one to store values and another to store 'square' of the values so these can later be summed individually. """

    s_block[tid_y, tid_x, 0] = s_thread       # First 2D array: store thread values
    s_block[tid_y, tid_x, 1] = s_thread**2    # Second 2D array: store 'square' of the thread values
    
    cuda.syncthreads()          # Synchronise Threads: WAIT UNTIL all threads store their values in the shared array


    # To Parallelise the Sum of all Shared array (each 2D) values:
    """We start with 8 threads working & STRIDE = 8, the first will sum values in s_block[0] and s_block[8]. The second, values in s_block[1] and s_block[9], until the last thread which will sum values s_block[7] and s_block[15].
      On the next step, only the first 4 threads need to work with STRIDE = 4. The first thread will sum s_block[0] and s_block[4]; the second, s_block[1] and s_block[5]; the third, s_block[2] and s_block[6]; the fourth and last, s_block[3] and s_block[7].
      On the third step, we now only need 2 threads with STRIDE = 2, to take care of the first 4 elements of s_block. The fourth and final step will use one thread with STRIDE = 1 to sum 2 elements. """
    i_x = cuda.blockDim.x // 2    # Initial STRIDE = 16/2 = 8
    i_y = cuda.blockDim.y // 2

    while (i_x != 0 and i_y != 0):
        if ((tid_x < i_x) and (tid_y < i_y)):
            s_block[tid_y, tid_x, 0] += s_block[tid_y+i_y, tid_x + i_x, 0]    # Summing 'values' in first 2D shared array
            s_block[tid_y, tid_x, 1] += s_block[tid_y+i_y, tid_x + i_x, 1]    # Summing 'Square of values' in second 2D shared array

        cuda.syncthreads()   # Wait until all values from other part of the shared array are added to these parts of the array, then repeat after REDUCING the STRIDE
        i_x //= 2      # Reduce STRIDE by half for next cycle
        i_y //= 2
    
    """ All sum values has been stored in the FIRST ELEMENT of the each of two 2D Shared array. """

    if (tid_x == 0 and tid_y == 0):
        #psr[bid] = s_block[0, 0]
        s_block[0, 0, 0] = s_block[0, 0, 0] * 1/(kernel_size*kernel_size)    # MEAN = Sum of values / size of kernel = (sum / 11*11) : stored in (0,0) of first 2D array
        s_block[0, 0, 1] = math.sqrt(s_block[0, 0, 1]* 1/(kernel_size*kernel_size) - math.pow(s_block[0, 0, 0], 2))   # VARIANCE = mean of square values - square of mean value : sotred in (0,0) of second 2D array

        """ PSR = (maxima value - mean) / std """
        psr[bid] = (inp[k_center_y, k_center_x] - s_block[0, 0, 0]) / s_block[0, 0, 1]  # Storing PSR Value from each Block (corresponding to each maxima location) in PSR array
            
        
        

CO_gpu = cuda.to_device(CO)     # Copy variable to GPU
CO_maxima_array_gpu = cuda.to_device(CO_maxima_array)

kernel_size = (11, 11)

psr_array_cpu = np.zeros((number_of_maxima), dtype=np.float32)
psr_array_gpu = cuda.to_device(psr_array_cpu)

maxima_mean_std[blocks_per_grid, threads_per_block_2d](CO_gpu, CO_maxima_array_gpu, kernel_size[0], psr_array_gpu)

psr_array = psr_array_gpu.copy_to_host()

# print("psr array: ", psr_array)




# from time import perf_counter_ns

# # Compile and then clear GPU from tasks
# maxima_mean_std[blocks_per_grid, threads_per_block_2d](CO_gpu, CO_maxima_array_gpu, kernel_size[0], psr_array_gpu)
# cuda.synchronize()

# timing = np.empty(31)
# for i in range(timing.size):
#     tic = perf_counter_ns()
#     maxima_mean_std[blocks_per_grid, threads_per_block_2d](CO_gpu, CO_maxima_array_gpu, kernel_size[0], psr_array_gpu)
#     cuda.synchronize()
#     toc = perf_counter_ns()
#     timing[i] = toc - tic
# timing *= 1e-3  # convert to μs

# print(f"Elapsed time: {timing.mean():.0f} ± {timing.std():.0f} μs")


# " Elapsed time: 134 ± 8 μs " # with 12, 12 thread per block
" GPU time: 144 ± 8 μs " # with 16, 16 thread per block



# Maxima Location:
""" Corresponds to the maxima candidate location which has Highest PSR (Peak to Side Lobe Ratio)"""
print(max(psr_array))
max_positions = np.argmax(psr_array)
print('max positons: ', max_positions)


max_coords = CO_maxima_array[max_positions]     # ** MAXIMA COORDINATES ** in image 'CO'
print('Maxima Coordinates: ', max_coords)






# ------------------------------------------------------------------------------------------------------------

# Part - 3
# Iris Boundary Refinement:

def gradient(iris_image_smooth, coordinate):
    S_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    S_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    local_region = iris_image_smooth[coordinate[0]-1:coordinate[0]+2, coordinate[1]-1:coordinate[1]+2]

    # Compute the gradient in the x and y directions
    gradient_x = abs(np.sum(local_region * S_x))
    gradient_y = abs(np.sum(local_region * S_y))

    return gradient_y, gradient_x



angle_step_size = 0.04      # Number of Angle Steps = 2*3.14 / 0.04 = (157 Steps) : means 2*pi to be covered in 157 steps

def subpixel_edge_refining(iris_image_smooth, max_coords, R_min, R_max, angle_step_size, grad_threshold):  # add (agree_threshold)
    n_max = int(2*3.14 / angle_step_size)   # (= 157)  for 0.04 step size

    candidate_points = np.zeros((n_max, 2), dtype= np.int32)
    iter = 0

    # max_grad_mag_array = np.zeros((n_max, 1), dtype= np.int32)       # temp test
    # max_grad_angle_array = np.zeros((n_max, 1), dtype= np.int32)     # temp test
    points = np.zeros((n_max, 2), dtype= np.int32)                   # temp test

    # Loop over angles:
    for theta in np.arange(0, 2 * 3.14, angle_step_size):
        temp_best_coord = np.array([0, 0], dtype=np.int32)    # (y, x)
        temp_best_prop = np.array([0, -100])   # [normalized_gradient_mag, normalized_gradient_angle] : put two random high values: put according to possible highest value

        # Loop over radii
        for r in range(R_min, R_max + 1):
            # Calculate Pt
            pt = np.array([int(max_coords[0] + r * np.sin(theta)), int(max_coords[1] +  r * np.cos(theta))])

            # Calculate gradients and magnitude at the points
            gy, gx = gradient(iris_image_smooth, pt)   # gradient function
            gradient_magnitude = np.sqrt(gx**2 + gy**2)

            # Calculate normalized gradient vector
            normalized_gradient = np.array([gy/gradient_magnitude, gx/gradient_magnitude], dtype=np.float32)
            normalized_gradient_mag = np.sqrt(normalized_gradient[0]**2 + normalized_gradient[1]**2)

            # Check threshold
            if normalized_gradient_mag < grad_threshold:
                continue
            else:
                # Calculate dot product of gradient magnitude and gradient direction = GRADIENT ANGLE
                normalized_gradient_angle = normalized_gradient[0] * (r*np.sin(theta)) + normalized_gradient[1] * (r*np.cos(theta))

                # Check cos(theta) * normalized_gradient > 'Agreement threshold' 
                # if normalized_gradient_angle > agree_threshold:
                    # Update temp_best if conditions met
                if (normalized_gradient_mag >= temp_best_prop[0] and normalized_gradient_angle >= temp_best_prop[1]):
                    temp_best_coord = pt
                    temp_best_prop = np.array([normalized_gradient_mag, normalized_gradient_angle])

        # Append temp_best_coord to CandidatePoints
        candidate_points[iter] = temp_best_coord
        iter += 1
    
    return candidate_points


grad_threshold = 0.8


# CPU Timing:
start_time_3 = time.time()   # gives in 'seconds'
points =  subpixel_edge_refining(iris_image_smooth, max_coords, R_min, R_max, angle_step_size, grad_threshold)   # add (agree_threshold)
end_time_3 = time.time()
    
print("CPU Time: ", (end_time_3-start_time_3) * 10**3, "ms")

" CPU Time:  61.2 ms "

# print('max_grad_mag_array: ', max_grad_mag_array)
# print('max_grad_angle_array: ', max_grad_angle_array)

#print('points: ', points)

color = 200  # BGR color (green in this example)
iris_image_smooth[points[:, 0], points[:, 1]] = color


# Display the image
fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.imshow(iris_image_smooth)
ax.set_title('Output Eye Image')
plt.show()




# ---------------------------------------------------------------------------------------------------------------------------------
# Part - 4
# Median Filter : Distance from center vs Angle

# 4 - (a):
@cuda.jit
def distance_plot(max_coords, points, points_dis_gpu):
    
    i = cuda.grid(1)   # thread index
    
    n0 = len(points)      # The last index is the fastest dimension

    if (i < n0):
        points_dis_gpu[i] = math.sqrt((points[i][0]-max_coords[0])**2 + (points[i][1]-max_coords[1])**2)



threads_per_block = 256    # 256 threads total per block
blocks_per_grid = 1    

max_coords_gpu = cuda.to_device(max_coords)
points_gpu = cuda.to_device(points)

points_dis = np.zeros((len(points)), dtype= np.float32)
points_dis_gpu = cuda.to_device(points_dis)

distance_plot[blocks_per_grid, threads_per_block](max_coords_gpu, points_gpu, points_dis_gpu)

points_distance = points_dis_gpu.copy_to_host()    # DISTANCE VALUES of the POINTS  (from the center maxima)

#print('points distances: ', points_distance)



# 4 - (b): ---------------------------------------------------------------
# Median Filter - (Version 1):

# window_size = 21               # Window Size of Median Filter
# window_size_const = int(21)         # Window Size of Median Filter (Constant version to be used in numba function to create fixed size local arrays)

# threads_per_block = 256    # 256 threads total per block
# blocks_per_grid = 1  

# @cuda.jit
# def median_filter_kernel(points_distance, filtered_points_distance, window_size):
#     i = cuda.grid(1)
#     n0 = len(points_distance)     # The last index is the fastest dimension
    
#     if (i< n0):
#         if (i < window_size // 2):
#             window = cuda.local.array(window_size_const, numba.float32)
#             window[0:(i + window_size // 2 + 1)] = points_distance[0:(i + window_size // 2 + 1)]
#             print(window)
#             # filtered_points_distance[i] = np.median(np.array([1, 2, 3]))
#         elif (i >= (n0 - window_size // 2)):
#             window = cuda.local.array(window_size_const, numba.float32)
#             window[0:(i + window_size // 2 + 1)] = points_distance[(i - window_size // 2):n0]
#             print(window)
#         else:
#             window = cuda.local.array(window_size_const, numba.float32)
#             window = points_distance[(i - window_size // 2):(i + window_size // 2 + 1)]
#             print(window)
#             # filtered_points_distance[i] = np.median(np.array([1, 2, 3]))



# Median Filter - (Version 2): ----------------------------------------------------------------

# threads_per_block = 32     # 32 threads (since Window Size = 21 used)
# blocks_per_grid = len(points_distance)     # = Number of Considered Maximas 

# @cuda.jit
# def median_filter_kernel(points_distance, filtered_points_distance, window_size, sorted_s_block):
    
#     bid = cuda.blockIdx.x       # Block ID
#     tid = cuda.threadIdx.x       # Thread ID X

#     n0 = len(points_distance)     # The last index is the fastest dimension
    
#     if (bid < window_size // 2):
#         if (tid <= bid + window_size // 2):  
#             value_thread = points_distance[tid]

#     elif (bid >= (n0 - window_size // 2)):
#         if (tid < n0 - bid + window_size // 2): 
#             value_thread = points_distance[bid - (window_size//2 - tid)]

#     else:
#         if (tid < window_size):
#             value_thread = points_distance[bid - (window_size//2 - tid)]


#     # Allocate shared array
#     s_block = cuda.shared.array(threads_per_block, numba.float32)   

#     s_block[tid] = value_thread       # Store thread values
    
#     cuda.syncthreads()          # Synchronise Threads: WAIT UNTIL all threads store their values in the shared array


#     if (tid == 0 and bid==0):  # Only first thread will find the MEDIAN of the contents of the Shared Array s_block
#         #block_median = np.nanmedian(s_block)   # NAN-MEDIAN: finds median with ignoring the NAN values
#         # result = np.zeros(np.shape(s_block), dtype=np.float64)
#         # result = np.sort(s_block)
        
#         print(s_block)
        
        #filtered_points_distance[bid] = block_median        
        

  

# points_distance_gpu = cuda.to_device(points_distance)

# filtered_points_dist = np.zeros(len(points_distance), dtype= np.float32)
# filtered_points_distance_gpu = cuda.to_device(filtered_points_dist)

# sorted_each = np.zeros(threads_per_block, dtype= np.float32)

# window_size = 21               # Window Size of Median Filter

# median_filter_kernel[blocks_per_grid, threads_per_block](points_distance_gpu, filtered_points_distance_gpu, window_size, sorted_each)

# filtered_points_distance = filtered_points_distance_gpu.copy_to_host()    # MEDIAN FILTERED Distance values of the POINTS (from the center maxima)

# print('MEDIAN FILTERED Points Distances: ', filtered_points_distance)





# Median Filter - (Version 3): CPU Normal iteratively using NP.MEDIAN() --------------------------------------------------------------------------------------------------

# window_size = 21               # Window Size of Median Filter

# def median_filter_cpu(points_distance, window_size):
#     n = len(points_distance)
#     medians = np.zeros(len(points_distance), dtype=np.float32)
#     #means = np.zeros(len(points_distance), dtype=np.float32)
#     for i in range(n):
#         if (i < window_size // 2):
#             medians[i] = np.median(points_distance[0:(i + window_size // 2 + 1)])
#             #means[i] = np.mean(points_distance[0:(i + window_size // 2 + 1)])

#         elif (i >= (n - window_size // 2)):
#             medians[i] = np.median(points_distance[(i - window_size // 2):n])
#             #means[i] = np.mean(points_distance[(i - window_size // 2):n])

#         else:
#             medians[i] = np.median(points_distance[(i - window_size // 2):(i + window_size // 2 + 1)])
#             #means[i] = np.mean(points_distance[(i - window_size // 2):(i + window_size // 2 + 1)])
    
#     return medians


# # CPU Timing:
# start_time_3 = time.time()   # gives in 'seconds'
# medians = median_filter_cpu(points_distance, window_size)
# end_time_3 = time.time()
    
# print("CPU Time: ", (end_time_3-start_time_3) * 10**3, "ms")

" CPU Time:  4 ms"


# fig3 = plt.figure()
# ax = fig3.add_subplot(111)
# ax.plot(points_distance,'yo-')
# ax.plot(medians,'go-')
# ax.plot(means,'bo-')
# ax.plot(np.mean(means),'ro-')
# ax.set_title('Points and its median version')
# plt.show()



# Median Filter - (Version 4): using SCIPY Library:-----------------------------------------

import scipy as sp
from scipy import signal, ndimage


# # CPU Timing:
# start_time_3 = time.time()   # gives in 'seconds'
# medians_scipy = ndimage.median_filter(points_distance, 21, mode='reflect')         # median filter by scipy library
# end_time_3 = time.time()
    
# print("CPU Time: ", (end_time_3-start_time_3) * 10**3, "ms")

window_size = 21          # Window Size of Median Filter

#medians_scipy = sp.signal.medfilt(points_distance,21) # median filter by scipy library
medians_scipy = ndimage.median_filter(points_distance, window_size, mode='reflect')         # median filter by scipy library

" CPU Time: 1 ms"


def filter_points (points, points_distance, medians, median_threshold):
    points_filtered = np.zeros((len(points_distance), 2), dtype=np.int32)
    for i in range(len(points_distance)):
        if (medians[i]+median_threshold >= points_distance[i] >= medians[i]-median_threshold ):
            points_filtered[i] = points[i]
    return points_filtered



median_threshold = 1       # Points whose distance form center maxima is beyond Median +/- 1 will be eliminated

points_filtered = filter_points (points, points_distance, medians_scipy, median_threshold)   # FINAL MEDIAN FILTERED POINTS  # with [0, 0] for Non-Selected
# (has [0, 0] for positions which are Not Selected: filtered out)

points_filtered_final = points_filtered[points_filtered[:, 0]!=0]    # [0,0] REMOVED
#print(points_filtered)


# # CPU Timing:
# start_time_3 = time.time()   # gives in 'seconds'
# points_filtered = filter_points (points, points_distance, medians_scipy, median_threshold)     # FINAL MEDIAN FILTERED POINTS
# end_time_3 = time.time()
    
# print("CPU Time: ", (end_time_3-start_time_3) * 10**3, "ms")
" CPU Time: 1 ms "

# plot the results
fig3 = plt.figure()
ax1 = fig3.add_subplot(211)
ax1.plot(points_distance,'yo-')
ax1.plot(medians_scipy,'go-')
ax1.set_title('Points(yellow) | Medians (Green)')

ax2 = fig3.add_subplot(212)
ax2.plot(points_distance[points_filtered[:, 0]!=0] , 'ro-')
ax2.plot(np.mean(points_distance[points_filtered[:, 0]!=0]) , 'go-')
ax2.set_title('Filtered Points (Red) | Mean (Green Dot)')
plt.show()





# ------------------------------------------------------------------------------------------------------------------------
# Part 5 : RANSAC + Curve Fitting:


from skimage.measure import EllipseModel, ransac

ransac_model, inliers = ransac(points_filtered_final, EllipseModel, min_samples=len(points_filtered_final), residual_threshold=3, max_trials=1)
" Not doing Ransac as of now, using all the points, so min_samples=len(points_filtered_final) "

ellipse_params = abs(np.int32(ransac_model.params))   # Ellipse Parameters: center_y, center_x, major_axis, minor_axis, angle
                                                      # Inliers:  points which are not far from the ellipse by more than the given threshold

# # CPU Timing:
# start_time_3 = time.time()   # gives in 'seconds'
# ransac_model, inliers = ransac(points_filtered_final, EllipseModel, min_samples=len(points_filtered_final), residual_threshold=3, max_trials=1)
# end_time_3 = time.time()
# ellipse_params = abs(np.int32(ransac_model.params))   
# print("CPU Time: ", (end_time_3-start_time_3) * 10**3, "ms")

" CPU Time: 25 ms ( if 5-10 points taken in Ransac)"
" CPU Time: 5 ms ( All points taken in Ransac)"


print('final ellipse params: ', abs(np.int32(ransac_model.params)))
print('Inliers: ', inliers)

print(type(ellipse_params[0]))

from skimage.draw import ellipse_perimeter

# Drawing Ellipse on the Image using the obtained ellipse Parameters:
rr, cc = ellipse_perimeter(ellipse_params[0], ellipse_params[1], ellipse_params[2], ellipse_params[3], ellipse_params[4])
iris_image_smooth[rr, cc] = 200

ellipse_center = [np.mean(rr), np.mean(cc)]     #  ** FINAL EYE PUPIL CENTER **

print('Ellipse Center: ', ellipse_center)

# Draw a circle of red color of thickness -1 px 
iris_image_smooth = cv.circle(iris_image_smooth, (np.int32(ellipse_center[1]), np.int32(ellipse_center[0])), 3, 200, 4)

# Display the image
fig4 = plt.figure()
ax = fig4.add_subplot(111)
ax.imshow(iris_image_smooth)
ax.set_title('Eye Image with Fitted Ellipse')
plt.show()









































