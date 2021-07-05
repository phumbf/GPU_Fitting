'''
Python wrapper to generate the fit data and then make a direct call to the C++ compiled 
dynamic link library (dll).

An array of Gaussians are generated and then flattened for the gpufit functionality.

The python "ctypes" module is used to provide an interface between the python and C++ code.
'''
import numpy as np
import ctypes
import matplotlib.pyplot as plt

#For plotting an example gaussian
def plot_example(x,y,params):

    #Plot toy data
    plt.plot(x,y,"o",label="Toy Data")

    #Plot the resultant fit
    x_vals = np.linspace(0,70,100)
    amp = params[0]
    mu = params[1]
    sig = params[2]
    off = params[3]

    plt.plot(x_vals,gaussian(x_vals,amp,mu,sig,off),label="GPU Fit",color="r")
    plt.legend()
    plt.tight_layout()
    plt.show()

#Define the gaussian
def gaussian(xvals,a0,a1,a2,a3):
    return (a0*np.exp(-(xvals-a1)**2/ (2*a2**2))) + a3

#Define the height and width of the array of gaussians
height = 1000
width = 1000

#Define the number of x-coords per fit - this is fixed for every fit
x = np.linspace(0,70,100)
n_x = len(x)

#Define the original mean, sigma, amplitude and offset
mu = 35
sig = 5
amp = 10
off = 0

#Create the matrix for x and y vals
y_mat = np.ndarray(shape=(height,width,n_x))

#Create array of starting fit values
f_mat = np.ndarray(shape=(height,width,4))

#Fill the matrix
for h in range(height):
    for w in range(width):

        #Fluctuate the parameters for generating the different data
        mu_fit = np.random.poisson(mu)
        sig_fit = np.random.poisson(sig)
        amp_fit = np.random.poisson(amp)
        off_fit = np.random.poisson(off)
        
        y = gaussian(x,amp_fit,mu_fit,sig_fit,off_fit)
        y_mat[h][w][:] = y
        f_mat[h][w][:] = np.array([amp,mu,sig,off])

#Flatten the matrices
f_x_mat = x
f_y_mat = y_mat.reshape(height*width*n_x)
f_f_mat = f_mat.reshape(height*width*4)

#Create result matrices for params, chisq, num iter and fit status
f_r_mat = np.zeros_like(f_f_mat)
f_c_mat = np.zeros(height*width)
f_i_mat = np.zeros(height*width)
f_s_mat = np.zeros(height*width)

#Prepare for C++ call using ctypes
c_x_mat = np.ctypeslib.as_ctypes(f_x_mat)
c_y_mat = np.ctypeslib.as_ctypes(f_y_mat)
c_f_mat = np.ctypeslib.as_ctypes(f_f_mat)
c_r_mat = np.ctypeslib.as_ctypes(f_r_mat)
c_c_mat = np.ctypeslib.as_ctypes(f_c_mat)
c_i_mat = np.ctypeslib.as_ctypes(f_i_mat)
c_s_mat = np.ctypeslib.as_ctypes(f_s_mat)

#Create ctype int variables
c_n_x = ctypes.c_int(n_x)
c_w = ctypes.c_int(width)
c_h = ctypes.c_int(height)

#Load in the .dll and run C++ fit
cdll = ctypes.CDLL("C:\\Users\\EMillard\\Documents\\Visual Studio 2015\\Projects\\GPU_Fitting\\x64\\Debug\\GPU_Fit.dll")

cdll.runGpuFit(c_y_mat,
                   c_x_mat,
                   c_f_mat,
                   c_r_mat,
                   c_c_mat,
                   c_i_mat,
                   c_s_mat,
                   c_n_x,
                   c_w,
                   c_h)

#Convert the final params back to a numpy array
r_p = np.ctypeslib.as_array(c_r_mat)
r_c = np.ctypeslib.as_array(c_c_mat)
r_i = np.ctypeslib.as_array(c_i_mat)
r_s = np.ctypeslib.as_array(c_s_mat)

#Get some stats of how the fits performed
tot_converged = 0
for f in range(height*width):
    if r_s[f] == 0:
        tot_converged += 1

print("Percentage of fits which converged = ",100*tot_converged/(height*width),"%")

#Unflatten the arrays - useful to probe specific fit
u_p = r_p.reshape(height,width,4)
u_c = r_c.reshape(height,width)
u_i = r_i.reshape(height,width)
u_s = r_s.reshape(height,width)

#Plot an example curve
test_height = 32
test_width = 23
print("The chisq was",u_c[test_height][test_width])
print("The number of iterations was",u_i[test_height][test_width])
print("The fit status was",u_s[test_height][test_width])
plot_example(x,
             y_mat[test_height][test_width],
             u_p[test_height][test_width])
