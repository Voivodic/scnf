import os
import numpy as np
import h5py as h5
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import NN
from tqdm import tqdm

#Set the percentage of VRAM pre-allocated by JAX (default = 0.75)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

#Define tyhe arrays to be used
Nfiles = 6
Ngrids = 500
Nsims = Nfiles*Ngrids
Ntrain = int(4*Ngrids)
Nvalidation = int(2*Ngrids)
Nd = 128
Lambda = [0.35, 0.20]

#Define the arrays for the theta and data
As = np.zeros(Nsims, dtype = "float32")
h = np.zeros(Nsims, dtype = "float32")
omega_cdm = np.zeros(Nsims, dtype = "float32")
data = np.zeros([Nsims, Nd, Nd, Nd], dtype = "float32")

#Define the limits for each parameter
f = h5.File("Simulations4/Grids_linear_0.h5", "r")
theta_lim = np.array(f["theta_lim"])
f.close()

#Run over all files and take the data
print("Reading the grids")
for i in tqdm(range(Nfiles)):
    f = h5.File("Simulations4/Grids_linear_%d.h5" %(i), "r")
    As[i*Ngrids : (i+1)*Ngrids] = f["As"]
    h[i*Ngrids : (i+1)*Ngrids] = f["h"]
    omega_cdm[i*Ngrids : (i+1)*Ngrids] = f["omega_cdm"]

    #data[i*Ngrids : (i+1)*Ngrids, 0,  :] = np.array([f["Grid_Lambda%.2f" %(Lambda[0])]["Grid_%d" %(j)] for j in range(Ngrids)], dtype = "float32")
    #data[i*Ngrids : (i+1)*Ngrids, 1,  :] = np.array([f["Grid_Lambda%.2f" %(Lambda[1])]["Grid_%d" %(j)] for j in range(Ngrids)], dtype = "float32")
    data[i*Ngrids : (i+1)*Ngrids, :] = np.array([f["Grid_Lambda%.2f" %(Lambda[0])]["Grid_%d" %(j)] for j in range(Ngrids)], dtype = "float32")

    f.close()

print(data.shape)

#Combine the parameters in the same array
theta = np.transpose(np.vstack([As, h, omega_cdm]))
#theta = np.transpose(np.vstack([As]))

#Get a subsample of the simulations
print("Preparing theta and data")

#Normalize the data and theta
mean_data = np.mean(data[:Ntrain,:], axis = 0)
std_data = np.std(data[:Ntrain,:], axis = 0)
mean_theta = np.mean(theta[:Ntrain,:], axis = 0)
std_theta = np.std(theta[:Ntrain,:], axis = 0)

data = (data - mean_data)/std_data
theta = (theta - mean_theta)/std_theta

#Set the hyperparameters
NDIM_PARAM = theta.shape[-1]
NDIM_DATA = data.shape[1:]
Nchannels = 1
DATASET_SIZE = Ntrain
BATCH_SIZE = 40
N_BATCHES = int(DATASET_SIZE/BATCH_SIZE)
N_BLOCKS = 1
N_NEURONS = [32, 32]
N_NEUROS_MS = [16]
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
N_STEPS = 1000
compact_mult = 2
PRINT_EVERY = 1
SEED = 12345
suffix = "CNF_N%d_grid_equivariant_beta10_latent%d" %(Nsims, compact_mult)

#Compute the weights for each parameter
theta_lim = np.array([[2.45e-09, 6.00e-01, 5.00e-02], [2.75e-09, 7.00e-01, 1.50e-01]])
theta_center = (theta_lim[1,:] + theta_lim[0,:])
delta_theta = theta_lim[1,:] - theta_lim[0,:]
weight_params = theta_center/delta_theta
weight_params = jnp.array(weight_params/np.mean(weight_params)/compact_mult)
#weight_params = jnp.ones(3)
print(weight_params)

#Create the output folder
try:
    os.system("mkdir Outputs/")
except:
    pass

#Save some informations
f = h5.File("Outputs/Infos_%s.h5" %(suffix), "w")
f.create_dataset("theta_lim", data = theta_lim)
f.create_dataset("mean_data", data = mean_data)
f.create_dataset("std_data", data = std_data)
f.create_dataset("mean_theta", data = mean_theta)
f.create_dataset("std_theta", data = std_theta)
f.close()

#Set the keys used by jax
key = jrandom.PRNGKey(SEED)
key_inf, key_train = jrandom.split(key, 2)

#Set the optimizator
optim = optax.adamw(learning_rate = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

#Define the loader function for y
def loader_y(inds):
    return jnp.array(theta[inds,:])

#Define the loader function for x
def loader_x(inds, shift = [], rotation = []):
    out = data[inds,:]

    #Rotate the grid
    if(len(rotation) > 0):
        axeses = [(0,1), (0,2), (1,2)]
        if(Nchannels == 1):
            for i in range(out.shape[0]):
                for j in range(3):
                    out[i,:,:,:] = np.rot90(out[i,:,:,:], k = rotation[i, j], axes = axeses[j])
        else:
            for i in range(out.shape[0]):
                for j in range(3):
                    for k in range(Nchannels):
                        out[i,k,:,:,:] = np.rot90(out[i,k,:,:,:], k = rotation[i, j], axes = axeses[j])

    #Transform to jax
    out = jnp.array(out)

    #Select the axis for the shifts
    if(Nchannels == 1):
        axis_shift = (0, 1, 2)
    else:
        axis_shift = (1, 2, 3)

    #Shift function
    def Shift(grid, shift):
        return jnp.roll(grid, shift = shift, axis = axis_shift)

    # Shift the grid
    if(len(shift) > 0):
        out = jax.vmap(Shift)(out, shift)

    return out

#Create the data dictionary
data_dict = {"Ntrain": Ntrain, "Nvalidation": Nvalidation, "theta_size": NDIM_PARAM, "data_size": NDIM_DATA, "theta_loader": loader_y, "data_loader": loader_x, "Nchannels": Nchannels}

#Run the analysis for the posterior PDF
print("Defining the inference class")
inf_posterior = NN.Inference(model = None, theta = None, data = data_dict, key = key_inf, num_blocks = N_BLOCKS, Nneurons = N_NEURONS, Nneurons_mean_std = N_NEUROS_MS, kind = "posterior", Use_conv = True, Nconv = 3, Nlin = 2, compacted_mult = compact_mult, momentum_size = 1, compact_type = "equivariant")
optim_state = None

#Train the CNF
print("Starting the training")
optim_state = inf_posterior.train(Nsteps = N_STEPS, Nbatches = N_BATCHES, print_every = PRINT_EVERY, optim = optim, key = key_train, optim_state = optim_state, Shift = True, Rotate = False, suffix = suffix, Ntimes = 5, poly_order = 1, alpha = 10.0, beta = 10.0, weight_params = weight_params, lr_limit = 1e-3)

print("Trained!")
