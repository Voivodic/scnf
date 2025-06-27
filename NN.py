#Import the libraries
from jaxtyping import Int, Float, Array, PRNGKeyArray
from typing import List, Union, Callable
from tqdm import tqdm
import h5py as h5
import os

import diffrax
import equinox as eqx 
import jax
import jaxlib
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import optax 
from optax import tree_utils as otu
import e3nn_jax as e3nn

#Class with one equivariant under E(3) layer
class E3conv_layer(eqx.Module):
    phi: List
    r_grid: Float[Array, "kernel_size*kernel_size*kernel_size"]
    weights: List
    kernels: List
    Kernel: list
    kernel_size: int
    stride: int
    irreps_in: e3nn.Irreps
    irreps_out: e3nn.Irreps
  
    #Initialize the class
    def __init__(self, irreps_in: e3nn.Irreps, irreps_out: e3nn.Irreps, key: PRNGKeyArray, kernel_size: int = 3, stride: int = 1, cell_size: int = 1.0):
        #Check if the kernel size is odd
        if(kernel_size % 2 == 0):
            raise ValueError("The kernel_size must be odd!")
        
        #Create the positions of the kernel
        self.kernel_size = kernel_size
        self.stride = stride
        kernel_side = (kernel_size - 1)/2
        x, y, z = jnp.meshgrid(jnp.arange(-kernel_side, kernel_side+1), jnp.arange(-kernel_side, kernel_side+1), jnp.arange(-kernel_side, kernel_side+1))
        pos_grid = jnp.stack((x, y, z), axis=-1)*cell_size
        self.r_grid = jnp.sqrt(jnp.sum(jnp.power(pos_grid, 2), axis = -1))
        
        #Split the key for the radial MLP and the weights of the kernels
        key_radial, key_angular = jrandom.split(key, 2)
        
        #Get informations about the input and output representations
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        ls_in = irreps_in.ls
        ls_out = irreps_out.ls
        Jmax = irreps_in.lmax + irreps_out.lmax
        if(Jmax > 3*kernel_side**2):
            Jmax = int(3*kernel_side**2)
                
        #Compute the spherical harmonics used
        YJ = []
        for i in range(Jmax+1):
            YJ.append(e3nn.sh(irreps_out = i, input = pos_grid, normalize = True))

        #Define the windows used to remove the high frequencies around the center of the kernel
        WJ = []
        for i in range(Jmax+1):
            window = jnp.ones((kernel_size, kernel_size, kernel_size))
            window = window.at[self.r_grid < jnp.sqrt(i)*cell_size].set(0.0)
            WJ.append(window)
        self.r_grid = self.r_grid.reshape([kernel_size**3, 1])

        #Compute Q times YJ for each possible J to be used  
        self.kernels = []
        self.weights = []
        for jin in ls_in:
            tmpk = []
            tmpw = []
            for jout in ls_out:
                ttmpk = []
                ttmpw = []
                for J in range(abs(jin - jout), jin+jout+1):  
                    if(J > Jmax):
                        continue

                    #Compute the Clebsch Gordan coefficients
                    cg = jnp.array(e3nn.clebsch_gordan(int(jin), int(jout), int(J)))*jnp.sqrt(2.0*J + 1.0)
                    
                    #Compute the outer product
                    ttmpk.append(jnp.einsum("jilmn,lmn->jilmn", jnp.einsum("ijk,lmnk->jilmn", cg, YJ[J]), WJ[J]))
                    
                    #Initialize the weight for this angular kernel
                    key_angular, key_weight = jrandom.split(key_angular, 2)
                    ttmpw.append(jrandom.normal(key_weight))
                    
                #Save the kernels and weights for this jout
                tmpk.append(jnp.array(ttmpk))
                tmpw.append(jnp.array(ttmpw))
                
            #Save the kernels and weights for this jin
            self.kernels.append(tmpk)
            self.weights.append(tmpw)
                        
        #Define the MPL for the radial part
        depth = 2
        width = 4
        keys_radial = jrandom.split(key_radial, depth+1)
        self.phi = [eqx.nn.Linear(1, width, key = keys_radial[0])]
        for i in range(depth-1):
            self.phi.append(eqx.nn.Linear(width, width, key = keys_radial[i+1]))
        self.phi.append(eqx.nn.Linear(width, 1, key = keys_radial[-1]))

        #Set the initial kernel
        self.Kernel = [0.0]

    #Pre-compute the kernel used in the convolutions
    def compute_kernel(self):

        #Compute the radial part
        phir = self.r_grid
        for layer in self.phi:
            phir = jax.nn.tanh(jax.vmap(layer)(phir))
        phir = phir.reshape([1,1,self.kernel_size, self.kernel_size, self.kernel_size])

        #Compute the angular part 
        kernel = jtu.tree_map_with_path(lambda _, x, y: jnp.einsum("a,aijlmn->ijlmn", x, y), self.weights, self.kernels)
        kernel = jnp.hstack([jnp.vstack(x) for x in kernel])

        #Construct the final kernel
        self.Kernel[0] = kernel*phir  
     
    #Compute the layer for a given input
    def __call__(self, x: e3nn.IrrepsArray):
        #Check the inputs
        if(x.irreps != self.irreps_in):
            raise ValueError("The irreps of the input is incompatile with the one used to initialize the class!")
        
        #Take the input array
        y = jnp.transpose(x.array, (3,0,1,2))

        #Compute the convolution of the kernel with the input array
        y = jax.lax.conv_general_dilated(lhs = jnp.expand_dims(y, axis=0), rhs = self.Kernel[0], window_strides = [self.stride, self.stride, self.stride], padding = "VALID")
        return e3nn.IrrepsArray(self.irreps_out, jnp.transpose(y[0], (1,2,3,0)))       

#Define a layer that concatenate the time with the data (random choice from FFJORD)
class Concat_Time(eqx.Module):
    transform_y: eqx.nn.Linear
    time_dilatation: eqx.nn.Linear
    time_shift: eqx.nn.Linear

    #Initialize the parameters of each sub-layer
    def __init__(self, y_size: Int, out_size: Int, key: PRNGKeyArray):
        key1, key2, key3 = jrandom.split(key, 3)

        self.transform_y = eqx.nn.Linear(y_size, out_size, key = key1)
        self.time_dilatation = eqx.nn.Linear(1, out_size, key = key2)
        self.time_shift = eqx.nn.Linear(1, out_size, use_bias = False, key = key3)

    #Return the final concatenation
    def __call__(self, t: Float, y: Float[Array, "in_size"]):
        return self.transform_y(y)*jnn.sigmoid(self.time_dilatation(t)) + self.time_shift(t)
    
#Define the layer that compress the information in x (x is assumed to have even dimensions)
class Compress_x(eqx.Module):
    convs: List
    lins: List
    Nchannels: Int

    #Initialize the parameters of the class
    def __init__(self, x_size: Int[Array, "x_dim"], out_size: Int, key: PRNGKeyArray, Nconv: Int = 2, Nlin: Int = 2, Nchannels: Int = 1):
        #Set the keys
        key_conv, key_lin = jrandom.split(key, 2)
        keys_conv = jrandom.split(key_conv, Nconv)
        keys_lin = jrandom.split(key_lin, Nlin + 1)
        
        #Set some hyperparameters of the compression (!!!only work well for kernel_size = 4 so far!!!)
        kernel_size = 4
        stride_size = kernel_size

        #Set the dimensionality of x 
        dim = len(x_size)

        #Check if all dimensions are even
        for i in range(dim):
            if(x_size[i] % 2 != 0):
                raise ValueError("The size of x along direction %d is not even!" %(i))

        #Define the current sizes of the data
        current_size = list(x_size)

        #Set the number of channels of the convolutional layers
        conv_channels = [Nchannels]
        for i in range(Nconv):
            conv_channels.append(4*conv_channels[-1])

        #Construct the convolutional layers
        self.convs = []
        for i in range(Nconv):
            #Compute the size of the padding
            padd_size = []
            for j in range(dim):
                if current_size[j] % 4 == 0:
                    padd_size.append(0)
                else:
                    padd_size.append(1)

            #Define the convolutional layer used in the compression
            self.convs.append(eqx.nn.Conv(num_spatial_dims = dim, in_channels = conv_channels[i], out_channels = conv_channels[i+1], kernel_size = kernel_size, stride = stride_size, padding = padd_size, key = keys_conv[i]))

            #Set the new sizes
            for j in range(dim):
                current_size[j] = (current_size[j] - kernel_size + 2*padd_size[j])//stride_size + 1

        #Compute the linearized dimension after the convolutional layers
        Ndim_lin = conv_channels[-1]
        for i in range(dim):
            Ndim_lin *= current_size[i]

        #Set the number of channels of the linear layers
        lin_channels = [Ndim_lin]
        for i in range(Nlin):
            lin_channels.append(lin_channels[-1]//4)

        #Construc the linear layers
        self.lins = []
        for i in range(Nlin):
            self.lins.append(eqx.nn.Linear(lin_channels[i], lin_channels[i+1], key = keys_lin[i]))
        self.lins.append(eqx.nn.Linear(lin_channels[-1], out_size, key = keys_lin[-1]))
        
        #Set the number of channels
        self.Nchannels = Nchannels
        
    #Return the compressed x
    def __call__(self, x: Float[Array , "x_size"]):
        #Expand x one dimension to take into account the channels
        if(self.Nchannels == 1):
            x = jnp.expand_dims(x, axis = 0)

        #Apply the conv layers
        for i in range(len(self.convs)):
            x = jnn.tanh(self.convs[i](x))

        #Flatten the data
        x = x.ravel()

        #Apply the linear layers
        for i in range(len(self.lins) - 1):
            x = jnn.tanh(self.lins[i](x))
        x = self.lins[-1](x)

        return x

#Define the layer that compress the information in x (x is assumed to have even dimensions)
class CompressEQ_x(eqx.Module):
    convs: List
    lins: List
    Nchannels: Int
    pool: eqx.nn.Pool
    pad_size: tuple

    #Initialize the parameters of the class
    def __init__(self, x_size: Int[Array, "x_dim"], out_size: Int, key: PRNGKeyArray, kernel_size: int = 3, stride: int = 1, Nconv: Int = 3, Nlin: Int = 2, Nchannels: Int = 1, cell_size: float = 1.0, irreps_hidden_conv: e3nn.Irreps = e3nn.Irreps("4x0e+1x1o+1x2e"), pooling_size: int = 2, Nchannels_conv: int = 64):
        #Set the keys
        key_conv, key_lin = jrandom.split(key, 2)
        keys_conv = jrandom.split(key_conv, Nconv + 1)
        keys_lin = jrandom.split(key_lin, Nlin + 1)

        #Set the dimensionality of x 
        dim = len(x_size)

        #Check if all dimensions are even
        for i in range(dim):
            if(x_size[i] % 2 != 0):
                raise ValueError("The size of x along direction %d is not even!" %(i))
        kernel_side = (kernel_size - 1)//2
        self.pad_size = ((kernel_side, kernel_side),(kernel_side, kernel_side),(kernel_side, kernel_side),(0,0))
            
        #Set the input and output irreps
        irreps_in = e3nn.Irreps("1x0e")
        irreps_out = Nchannels_conv*e3nn.Irreps("1x0e")

        #Set the number of channels of the convolutional layers
        '''conv_irreps_in = [irreps_in]
        conv_irreps_out = []
        for i in range(Nconv):
            conv_irreps_in.append(2**i*irreps_hidden_conv)
            conv_irreps_out.append((conv_irreps_in[-1] + ((conv_irreps_in[-1].filter(drop = "0e+0o")).num_irreps)*e3nn.Irreps("1x0e")).regroup())
        conv_irreps_out.append(irreps_out)'''
        #Setting the number of channels of the convolutional layers BY HAND
        conv_irreps_in = [irreps_in]
        conv_irreps_out = []
        if(Nconv > 0):
            conv_irreps_in.append((conv_irreps_in[-1] + e3nn.Irreps("5x0e")).regroup())
            conv_irreps_out.append((conv_irreps_in[-1] + e3nn.Irreps("0x0e")).regroup())
        if(Nconv > 1):
            conv_irreps_in.append((conv_irreps_in[-1] + e3nn.Irreps("6x0e+1x2e")).regroup())
            conv_irreps_out.append((conv_irreps_in[-1] + e3nn.Irreps("1x0e")).regroup())
        if(Nconv > 2):
            conv_irreps_in.append((conv_irreps_in[-1] + e3nn.Irreps("12x0e+1x1o+1x2e")).regroup())
            conv_irreps_out.append((conv_irreps_in[-1] + e3nn.Irreps("3x0e")).regroup())
        conv_irreps_out.append(irreps_out)

        #Construct the convolutional layers
        self.convs = []
        for i in range(Nconv+1):
            self.convs.append(E3conv_layer(irreps_in = conv_irreps_in[i], irreps_out = conv_irreps_out[i], kernel_size = kernel_size, stride = stride, key = keys_conv[i], cell_size = cell_size))

        #Set the number of channels of the linear layers
        alpha = int(jnp.power(Nchannels_conv/out_size, 1.0/(Nlin+1)))
        alpha = 2 if alpha < 2 else alpha
        lin_channels = [int(Nchannels_conv)]
        for i in range(Nlin):
            lin_channels.append(int(lin_channels[-1]//alpha))
        lin_channels.append(out_size)

        #Construc the linear layers
        self.lins = []
        for i in range(Nlin+1):
            self.lins.append(eqx.nn.Linear(lin_channels[i], lin_channels[i+1], key = keys_lin[i]))
        
        #Set the pooling layer
        self.pool = eqx.nn.AvgPool3d(kernel_size = pooling_size, stride = pooling_size, padding = 0)

        #Set the number of channels
        self.Nchannels = Nchannels

    #Pre-compute the kernels of all conv layers
    def compute_kernels(self):
        for conv in self.convs:
            conv.compute_kernel()
        
    #Return the compressed x
    def __call__(self, x: Float[Array , "x_size"]):
        #Expand x one dimension to take into account the channels and corvert to IrrepsArray
        if(self.Nchannels == 1):
            x = jnp.expand_dims(x, axis = -1)
        x_irreps = e3nn.Irreps("1x0e")

        #Apply the conv layers
        for conv in self.convs:
            x = e3nn.IrrepsArray(x_irreps, jnp.pad(x, pad_width = self.pad_size, mode = 'wrap'))
            x = conv(x)
            x = e3nn.gate(x, even_act = jax.nn.tanh, even_gate_act = jax.nn.tanh, normalize_act = True)
            x_irreps = x.irreps
            x = jnp.transpose(self.pool(jnp.transpose(x.array, (3,0,1,2))), (1,2,3,0))

        #Flatten the data
        x = jnp.mean(x, axis = (0,1,2))

        #Apply the linear layers
        for i in range(len(self.lins) - 1):
            x = jax.nn.relu(self.lins[i](x))
        x = self.lins[-1](x)
        
        return x
     
#Define a layer that concatenate the time and x with the data (random choice from FFJORD)
class Concat_Time_and_x(eqx.Module):
    concat_layer: eqx.nn.Linear
    time_dilatation: eqx.nn.Linear
    time_shift: eqx.nn.Linear
    compact_x: eqx.Module
    transformed_x: list
    momentum_size: int

    #Initialize the parameters of each sub-layer
    def __init__(self, y_size: Int, x_size: Union[Int, Int[Array, "x_dim"]], out_size: Int, key: PRNGKeyArray, Nconv: Int = 3, Nlin: Int = 2, compacted_size: Int = 0, momentum_size: Int = 1, Nchannels: Int = 1, compact_type: str = "standard"):
        key1, key2, key3, key4 = jrandom.split(key, 4)
        
        #Set the compact and momentum size
        if(compacted_size == 0):
            compacted_size = y_size
        self.momentum_size = momentum_size

        self.concat_layer = eqx.nn.Linear(y_size + compacted_size + momentum_size, out_size, key = key1)
        self.time_dilatation = eqx.nn.Linear(1, out_size, key = key2)
        self.time_shift = eqx.nn.Linear(1, out_size, use_bias = False, key = key3)

        if(type(x_size) == int):
            self.compact_x = eqx.nn.Linear(x_size, compacted_size, key = key4)
        else:
            if(compact_type == "standard"):
                self.compact_x = Compress_x(x_size, compacted_size, key = key4, Nconv = Nconv, Nlin = Nlin, Nchannels = Nchannels)
            elif(compact_type == "equivariant"):
                self.compact_x = CompressEQ_x(x_size = x_size, out_size = compacted_size, key = key4, kernel_size = 3, stride = 1, Nconv = Nconv, Nlin = Nlin, Nchannels = Nchannels, cell_size = 1.0, irreps_hidden_conv = e3nn.Irreps("4x0e+1x1o+1x2e"), pooling_size = 2, Nchannels_conv = 64)

        #Set the initial output of the layers with x
        self.transformed_x = [0.0]
        for _ in range(momentum_size):
            self.transformed_x.append(0.0)

    #Compute the x after transformed by the layes
    def transform_x(self, x: Float[Array, "x_size"]):
        self.transformed_x[0] = jnn.tanh(self.compact_x(x))
        for i in range(self.momentum_size):
            self.transformed_x[1+i] = jnp.mean(jnp.power(x, i+2))

    #Return the final concatenation
    def __call__(self, t: Float, y: Float[Array, "in_size"]):
        y_stacked = jnp.hstack([y, self.transformed_x[0], self.transformed_x[1:]])
        y = self.concat_layer(y_stacked)*jnn.sigmoid(self.time_dilatation(t)) + self.time_shift(t)

        return y
    
#Define the vector field (right hand side of the ODE)
class Linear_layer(eqx.Module):
    layers: List

    #Initialize the NN
    def __init__(self, y_size: Int, key: PRNGKeyArray, Nneurons: Int[Array, "Nlayers"] = [64, 64], x_size: Union[Int, Int[Array, "x_dim"]] = 0, Nconv: Int = 3, Nlin: Int = 2, compacted_size: Int = 0, momentum_size: Int = 1, Nchannels: Int = 1, compact_type: str = "standard"):
        Nlayers = len(Nneurons)        
        keys = jax.random.split(key, Nlayers + 1)

        #Define the layers used by the model non conditional
        if(x_size == 0):
            if(Nlayers == 0):
                self.layers = [Concat_Time(y_size, y_size, key = keys[0])]
            else:
                self.layers = [Concat_Time(y_size, Nneurons[0], key = keys[0])]
                for i in range(Nlayers - 1):
                    self.layers.append(Concat_Time(Nneurons[i], Nneurons[i+1], key = keys[i+1]))
                self.layers.append(Concat_Time(Nneurons[i], y_size, key = keys[-1]))
        
        #Define the layers used by the conditional model
        else:
            if(Nlayers == 0):
                self.layers = [Concat_Time_and_x(y_size, x_size, y_size, key = keys[0], Nconv = Nconv, Nlin = Nlin, compacted_size = compacted_size, momentum_size = momentum_size, Nchannels = Nchannels, compact_type = compact_type)]
            else:
                self.layers = [Concat_Time_and_x(y_size, x_size, Nneurons[0], key = keys[0], Nconv = Nconv, Nlin = Nlin, compacted_size = compacted_size, momentum_size = momentum_size, Nchannels = Nchannels, compact_type = compact_type)]
                for i in range(Nlayers - 1):
                    self.layers.append(Concat_Time_and_x(Nneurons[i], x_size, Nneurons[i+1], key = keys[i+1], Nconv = Nconv, Nlin = Nlin, compacted_size = compacted_size, momentum_size = momentum_size, Nchannels = Nchannels, compact_type = compact_type))
                self.layers.append(Concat_Time_and_x(Nneurons[i], x_size, y_size, key = keys[-1], Nconv = Nconv, Nlin = Nlin, compacted_size = compacted_size, momentum_size = momentum_size, Nchannels = Nchannels, compact_type = compact_type))          

    #Return the output of the NN
    def __call__(self, t: Float, y: Float[Array, "y_size"], args):
        #Convert the time to a jax array
        t = jnp.asarray(t)[None]

        #Run over all layers
        for layer in self.layers[:-1]:
            y = jnn.relu(layer(t, y))
        y = self.layers[-1](t, y)

        return y
    
#Define the function that maps x to the mean and std of the base distribution
class Mean_and_std(eqx.Module):
    layers_mu: List
    layers_std: List
    
    #Initialize the NN
    def __init__(self, y_size: Int, x_size: Union[Int, Int[Array, "x_dim"]], key: PRNGKeyArray, Nneurons: Int[Array, "Nlayers"] = [], Nconv: Int = 3, Nlin: Int = 2, Nchannels: Int = 1, compact_type: str = "standard"):
        Nlayers = len(Nneurons)  
        key_mu, key_std = jax.random.split(key, 2)      
        keys_mu = jax.random.split(key_mu, Nlayers + 1)
        keys_std = jax.random.split(key_std, Nlayers + 1)

        #Define the layers used by the model
        if(type(x_size) == int):
            if(Nlayers == 0):
                self.layers_mu = [eqx.nn.Linear(x_size, y_size, key = keys_mu[0])]
                self.layers_std = [eqx.nn.Linear(x_size, y_size, key = keys_std[0])]
            else:
                self.layers_mu = [eqx.nn.Linear(x_size, Nneurons[0], key = keys_mu[0])]
                self.layers_std = [eqx.nn.Linear(x_size, Nneurons[0], key = keys_std[0])]
                for i in range(Nlayers - 1):
                    self.layers_mu.append(eqx.nn.Linear(Nneurons[i], Nneurons[i+1], key = keys_mu[i+1]))
                    self.layers_std.append(eqx.nn.Linear(Nneurons[i], Nneurons[i+1], key = keys_std[i+1]))
                self.layers_mu.append(eqx.nn.Linear(Nneurons[-1], y_size, key = keys_mu[-1]))
                self.layers_std.append(eqx.nn.Linear(Nneurons[-1], y_size, key = keys_std[-1]))    
        else:
            if(compact_type == "standard"):
                self.layers_mu = [Compress_x(x_size, y_size, key = keys_mu[0], Nconv = Nconv, Nlin = Nlin, Nchannels = Nchannels)]
                self.layers_std = [Compress_x(x_size, y_size, key = keys_std[0], Nconv = Nconv, Nlin = Nlin, Nchannels = Nchannels)]
            elif(compact_type == "equivariant"):
                self.layers_mu = [CompressEQ_x(x_size = x_size, out_size = y_size, key = keys_mu[0], kernel_size = 3, stride = 1, Nconv = Nconv, Nlin = Nlin, Nchannels = Nchannels, cell_size = 1.0, irreps_hidden_conv = e3nn.Irreps("4x0e+1x1o+1x2e"), pooling_size = 2, Nchannels_conv = 64)]
                self.layers_std = [CompressEQ_x(x_size = x_size, out_size = y_size, key = keys_std[0], kernel_size = 3, stride = 1, Nconv = Nconv, Nlin = Nlin, Nchannels = Nchannels, cell_size = 1.0, irreps_hidden_conv = e3nn.Irreps("4x0e+1x1o+1x2e"), pooling_size = 2, Nchannels_conv = 64)]                

    #Return the mean and std for this x
    def __call__(self, x: Float[Array, "x_size"]):
        #Run over all layers for mu
        y_mean = x
        for layer in self.layers_mu:
            y_mean = jnn.tanh(layer(y_mean))

        #Run over all layers for std
        y_std = x
        for layer in self.layers_std:
            y_std = jnn.tanh(layer(y_std))
        y_std = jnp.exp(y_std)

        return y_mean, y_std        

#Define the class for the continuos normalizing flow to approximate P(y|x)
class CNF(eqx.Module):
    funcs: List
    y_size: Int
    x_size: Int
    t0: Float
    t1: Float
    dt0: Float
    Mean_std: Mean_and_std
    compact_type: str
    compacted_mult: int
    momentum_size: int

    #Initialize the CNF
    def __init__(self, y_size: Int, key: PRNGKeyArray, num_blocks: Int = 1, Nneurons: Int[Array, "Nlayers"] = [32, 32], Nneurons_mean_std: Int[Array, "Nlayers"] = [16], x_size: Union[Int, Int[Array, "x_dim"]] = 0, Nconv: Int = 3, Nlin: Int = 2, compacted_mult: Int = 1, momentum_size: Int = 0, Nchannels: Int = 1, compact_type: str = "standard"):
        #Save the information in the class
        self.y_size = y_size
        self.x_size = x_size
        self.t0 = 0.0
        self.t1 = 1.0
        self.dt0 = 0.1  
        self.compacted_mult = compacted_mult
        self.momentum_size = momentum_size

        #Define num_blocks different transformations
        Key_mean_std, key_blocks = jrandom.split(key, 2)
        keys = jrandom.split(key_blocks, num_blocks)

        #Initialize each ODE
        self.funcs = []
        for k in keys:
            self.funcs.append(Linear_layer(y_size = y_size, x_size = x_size, key = k, Nneurons = Nneurons, Nconv = Nconv, Nlin = Nlin, compacted_size = y_size*compacted_mult, momentum_size = momentum_size, Nchannels = Nchannels, compact_type = compact_type)) 

        #Initialize the NN to compute the mean and std of the latent space gaussian distribution
        if(type(x_size) == int):
            if(x_size > 0):  
                self.Mean_std = Mean_and_std(y_size = y_size, x_size = x_size, key = Key_mean_std, Nneurons = Nneurons_mean_std, Nconv = Nconv, Nlin = Nlin, Nchannels = Nchannels, compact_type = compact_type)
            else:
                self.Mean_std = None
        else: 
            self.Mean_std = Mean_and_std(y_size = y_size, x_size = x_size, key = Key_mean_std, Nneurons = Nneurons_mean_std, Nconv = Nconv, Nlin = Nlin, Nchannels = Nchannels, compact_type = compact_type)
            
        #Save the type of compactification of x
        self.compact_type = compact_type

    #Define the log of the normal (distribution in the latent space)
    def log_normal(self, y: Float[Array, "y_size"], x: Float[Array, "x_size"] = None):
        if(x == None):
            return -0.5 * (y.shape[-1]*jnp.log(2 * jnp.pi) + jnp.sum(y**2))
        else:
            mu, std = self.Mean_std(x)
            
            return -0.5 * (y.shape[-1]*jnp.log(2 * jnp.pi) + 2.0*jnp.sum(jnp.log(std), axis = -1) + jnp.sum((y - mu)**2/std**2, axis = -1))
                        
    #Wrapper function that compute the vector field together with the trace of the jacobian
    def Wrapper_Func_TrJac_approx(self, t: Float, y: Float[Array, "2 y_size"], args):
        y, _ = y

        #Eps is a normal random vector used in the Hutchinson's approximation
        eps, func = args

        #Compute the approximation of the trace Jacobian
        fn = lambda y: func(t, y, ())
        f, vjp_fn = jax.vjp(fn, y)
        (eps_dfdy,) = vjp_fn(eps)
        trjac = jnp.sum(eps_dfdy * eps) 

        return f, trjac

    #Runs backward-in-time to train the CNF.
    def get_logP(self, y: Float[Array, "y_size"], key: PRNGKeyArray, save_ts: Float[Array, "Ntimes"], x: Float[Array, "x_size"] = None):    
        #Define the points to get the output
        saveat = diffrax.SaveAt(ts = save_ts)

        #Use the wrapper defined above to set the ODEs
        term = diffrax.ODETerm(self.Wrapper_Func_TrJac_approx)

        #Define the solver and sample epsilon
        solver = diffrax.Tsit5()
        eps = jrandom.normal(key, y.shape)

        #Solve back-in-time for all blocks
        delta_log_likelihood = 0.0
        transformed = []
        for func in reversed(self.funcs):
            #Transform the current value of x
            if(x != None):
                for layer in func.layers:
                    layer.transform_x(x)
                    transformed.append(jnp.hstack(layer.transformed_x))

            #Solve the ODE
            y = (y, delta_log_likelihood)
            sol = diffrax.diffeqsolve(term, solver, self.t1, self.t0, -self.dt0, y, (eps, func), saveat = saveat, stepsize_controller = diffrax.PIDController(rtol = 1e-4, atol = 1e-4))
            y, delta_log_likelihood = sol.ys

        #Return the result added to the log normal
        return y, delta_log_likelihood[-1] + self.log_normal(y[-1,:], x), jnp.array(transformed)
    
    #Solve the ODE foward in time
    @eqx.filter_jit
    def solve_ODE(self, y: Float[Array, "y_size"], Ntimes: Int = 1):
        #Define the points to get the output
        save_ts = jnp.linspace(self.t0, self.t1, Ntimes - 1, endpoint = False)
        save_ts = jnp.hstack([save_ts, self.t1])
        saveat = diffrax.SaveAt(ts = save_ts)

        #Solve the blocks
        solver = diffrax.Tsit5()
        out = []
        for func in self.funcs:                  
            #Solve the ODE
            sol = diffrax.diffeqsolve(diffrax.ODETerm(func), solver, self.t0, self.t1, self.dt0, y, saveat = saveat, stepsize_controller = diffrax.PIDController(rtol = 1e-5, atol = 1e-5))
            out.append(sol.ys)
            y = sol.ys[-1]

        return jnp.concatenate(out)  

    #Runs forward-in-time to draw samples from the CNF.
    def sample(self, key: PRNGKeyArray, Nsamples: Int = 1000, x: Float[Array, "x_size"] = None, prior: Callable = lambda _: 1, max_prior: Float = 0.0, Nmax: int = 10_000, Ntimes: Int = 1):
        #Compute the kernels for the case of the equivariant compactification
        if(self.compact_type == "equivariant"):
            for func in self.funcs:
                for layer in func.layers:
                    layer.compact_x.compute_kernels()
            for layer in self.Mean_std.layers_mu:
                layer.compute_kernels()
            for layer in self.Mean_std.layers_std:
                layer.compute_kernels()
       
        #Set x for this evolution
        if(x is not None):
            for func in self.funcs:
                for layer in func.layers:
                    layer.transform_x(x)
            mu, std = self.Mean_std(x)
        else:
            mu, std = 0.0, 1.0
        
        #Run till Nsamples samples were created
        Nout = 0
        count = 0
        y_out = jnp.empty([0, Ntimes*len(self.funcs), self.y_size])
        while(Nout < Nsamples and count < Nmax):
            key, key_normal, key_uni = jrandom.split(key, 3)
            
            #Generate the normal random state of the latent space
            y_ini = jrandom.normal(key_normal, (Nsamples, self.y_size))*std + mu
                        
            #Evolve the ODE in time
            y = jax.vmap(self.solve_ODE, in_axes = (0, None))(y_ini, Ntimes)
            
            #Compute the normalized prior for each solution
            if(max_prior != 0.0):
                P_prior = jax.vmap(prior)(y[:,-1,:])/max_prior
            else:
                P_prior = jax.vmap(prior)(y[:,-1,:])
                P_prior = P_prior/jnp.max(P_prior)
            
            #Generate random number between 0 and 1 and define the mask
            eps = jrandom.uniform(key_uni, shape = (Nsamples,))
            mask = eps < P_prior
            y_out = jnp.vstack([y_out, y[mask, :]])
            Nout = y_out.shape[0]   
            
            count += 1    

        if(Ntimes == 1):
            return y_out[:Nsamples, 0, :]
        else:
            return y_out[:Nsamples,:]
        
    #Get the state of the class as a dictionay
    def __getstate__(self):
        return self.__dict__

    #Set the state of the class using a dictionary
    def __setstate__(self, state):
        self.__dict__.update(state)

#Define the loss function used in the training
def loss(diff_model: CNF, static_model: CNF, data: Float[Array, "Nbatch data_size"], key: PRNGKeyArray, save_ts: Float[Array, "Ntimes"], MatrixT: Float[Array, "Ntimes Ntimes"] = None, alpha: Float = 0.0, x: Float[Array, "x_size"] = None, compact_type = "standard", beta: Float = 0.0, weight_params: Float[Array, "data_size"] = None):
    #Combine the differentiable and the static parts of the model
    model = eqx.combine(diff_model, static_model)

    #Pre-compute the kernels used for the convolution in each layer
    if(compact_type == "equivariant"):
        for func in model.funcs:
            for layer in func.layers:
                layer.compact_x.compute_kernels()
        for layer in model.Mean_std.layers_mu:
            layer.compute_kernels()
        for layer in model.Mean_std.layers_std:
            layer.compute_kernels()
    
    #Solve backward the ODE to get the logP
    y, logP, compacted = jax.vmap(model.get_logP, in_axes = (0, None, None, 0))(data, key, save_ts, x)

    #Compute the loss for the polynomial regularization
    if(MatrixT is not None and alpha > 0.0):
        Lreg = jnp.mean(jnp.matmul(MatrixT, y)**2)
    else:
        Lreg = 0
        
    #Compute the loss for the compactification regularization
    if(beta > 0.0):
        Ndata = int(data.shape[-1])
        if(weight_params is None):
            weight_params = jnp.ones(Ndata)
            
        Lparam = 0
        for i in range(model.compacted_mult):
            Lparam += jnp.mean(jnp.power((compacted[:,:,i*Ndata:(i+1)*Ndata]/data[:,jnp.newaxis,:] - 1.0), 2.0)*weight_params)
        Lparam = Lparam/model.compacted_mult
    else:
        Lparam = 0.0
    
    return -jnp.mean(logP) + alpha*Lreg + beta*Lparam

#Function that return the static leafs of the model tree
def get_static(tree):
    out = [tree.compact_type]

    #Check the filtering parts of the NODE
    for func in tree.funcs:
        for layer in func.layers:
            #Make static the transformed value of x
            out.append(layer.transformed_x)
            
            #Stuff for the equivariant compression
            try:
                out.append(layer.compact_x.pad_size)
                out.append(layer.compact_x.pool)
                for conv in layer.compact_x.convs:
                    #Arrays of the E3conv layer
                    out.append(conv.r_grid)
                    out.append(conv.kernels)
                    out.append(conv.Kernel)
                    out.append(conv.irreps_in)
                    out.append(conv.irreps_out)
            except:
                continue

    #Filter for the NN that computes the mean of the latent distribution
    try:
        for layer in tree.Mean_std.layers_mu:
            #Stuff for the equivariant compression
            out.append(layer.pad_size)
            out.append(layer.pool)
            for conv in layer.convs:
                #Arrays of the E3conv layer
                out.append(conv.r_grid)
                out.append(conv.kernels)
                out.append(conv.Kernel)
                out.append(conv.irreps_in)
                out.append(conv.irreps_out)
    except:
        pass

    #Filter for the NN that computes the std of the latent distribution
    try: 
        for layer in tree.Mean_std.layers_std:
            #Stuff for the equivariant compression
            out.append(layer.pad_size)
            out.append(layer.pool)
            for conv in layer.convs:
                #Arrays of the E3conv layer
                out.append(conv.r_grid)
                out.append(conv.kernels)
                out.append(conv.Kernel)
                out.append(conv.irreps_in)
                out.append(conv.irreps_out)
    except:
        pass
    return out

#Function that updates the weights one step
@eqx.filter_jit
def make_step(loss, model: CNF, data: Float[Array, "Nbatch data_size"], optim: optax._src.base.GradientTransformationExtraArgs, key: PRNGKeyArray, save_ts: Float[Array, "Ntimes"], MatrixT: Float[Array, "Ntimes  Ntimes"] = None, alpha: Float = 0.0, optim_state: tuple = None, lr_schedule_state: tuple = None, x: Float[Array, "x_size"] = None, compact_type = "standard",  beta: Float = 0.0, weight_params: Float[Array, "data_size"] = None):
    #Split the model between the trainable and fixed parameters
    len_static = len(get_static(model))
    filter_spec = jtu.tree_map(lambda _: True, model)
    filter_spec = eqx.tree_at(get_static, filter_spec, replace = [False for _ in range(len_static)])
    diff_model, static_model = eqx.partition(pytree = model, filter_spec = filter_spec) 

    #Compute the loss and it gradient
    loss_value, grads = eqx.filter_value_and_grad(loss)(diff_model, static_model, data, key, save_ts, MatrixT, alpha, x, compact_type, beta, weight_params)
    
    #Update the state and the learning rate
    updates, optim_state = optim.update(grads, optim_state, diff_model)
    updates = otu.tree_scalar_mul(lr_schedule_state.scale, updates)

    #Apply the updates
    diff_model = eqx.apply_updates(diff_model, updates)
    
    #Combine the differentiable and the static parts of the model
    model = eqx.combine(diff_model, static_model)

    return model, loss_value, optim_state

#Define the class that run the inference
class Inference(eqx.Module):
    models: List
    Ntrain: Int
    Nvalidation: Int
    y_size: Int
    x_size: Union[Int, Int[Array, "x_dim"]]
    x: Float[Array, "Nsims x_size"]
    y: Float[Array, "Nsims y_size"]
    loss_best: List
    losses: Float[Array, "Nsteps"]
    losses_validation: Float[Array, "Nsteps"]
    data_loader_y: Callable
    data_loader_x: Callable
    lr_history: List
    compact_type: str

    #I had to define the model and loss_best as arrays because the eqx.Module is a frozen class#
    #I put loss and make_step outside the class because it speed up the computation by a factor of ~3 (I don't know why)#

    #Initialize the class used for the inference
    def __init__(self, data: Union[Float[Array, "Nsims Ndim_data"], dict], key: PRNGKeyArray, Nvalidation: Int = 0, model: CNF = None, num_blocks: Int = 1, Nneurons: Int[Array, "Nlayers"] = [64, 64], Nneurons_mean_std: Int[Array, "Nlayers"] = [16], Nconv: Int = 3, Nlin: Int = 2, compacted_mult: Int = 1, momentum_size: Int = 1, kind: str = "joint", theta: Float[Array, "Nsims Ndim_param"] = None, Use_conv: bool = False, Nchannels: Int = 1, compact_type: str = "standard"):   
        
        #Set the parameters of the data and theta in case the data was given as an array
        if(type(data) == jaxlib.xla_extension.ArrayImpl):
            #Check if the data and parameter vector were given to compute the conditional PDFs
            if((theta == None and kind == "posterior") or (theta == None and kind == "likelihood")):
                raise ValueError("You have to give a theta array to fit the conditional distributions\n")

            #Take the number of simulations
            Nsims = data.shape[0]
            if(Nsims <= Nvalidation):
                raise ValueError("The number of simulation is smaller than the size of validation set (no simulation for training)!")
            self.Ntrain = data.shape[0] - Nvalidation
            self.Nvalidation = Nvalidation

            #Take the dimensionality of the data
            Ndim = len(data.shape[1:])
            
            #Get data and theta
            if(theta != None):
                #Check if the first dimension of theta and data is equal
                if(theta.shape[0] != data.shape[0]):
                    raise ValueError("The first dimensions of theta and data must be the same! %d != %d\n" %(theta.shape[0], data.shape[0]))
                
                #Set the dimensions of data and theta, save them and create the loader functions
                if(kind == "joint"):
                    self.y_size = theta.shape[1] + data.shape[1]
                    self.x_size = 0
                    self.y = jnp.hstack([theta, data])
                    self.x = None
                    self.data_loader_y = lambda inds: self.y[inds, :]
                    self.data_loader_x = lambda inds: None

                elif(kind == "posterior"):
                    self.y_size = theta.shape[1]
                    if(Ndim == 1 and Use_conv == False):
                        self.x_size = data.shape[1]   
                    else:
                        self.x_size = data.shape[1:]
                    self.y = theta
                    self.x = data
                    self.data_loader_y = lambda inds: self.y[inds, :]
                    self.data_loader_x = lambda inds, shift = None, rotation = None: self.x[inds, :]

                elif(kind == "likelihood"):
                    self.y_sitransform_statee = data.shape[1]
                    self.x_size = theta.shape[1]
                    self.y = data
                    self.x = theta
                    self.data_loader_y = lambda inds: self.y[inds, :]
                    self.data_loader_x = lambda inds: self.x[inds, :]
                
            else:
                self.y_size = data.shape[1]
                self.x_size = 0
                self.y = data
                self.x = None
                self.data_loader_y = lambda inds: self.y[inds, :]
                self.data_loader_x = lambda inds: None

        #Set the parameters of the data and theta in case the data was given as an dictionary
        elif(type(data) == dict):
            if(not all(key in data for key in ["Ntrain", "data_size", "theta_size", "theta_loader", "data_loader"])):
                raise ValueError("The data dictionary must contain the number of simulations (Ntrain), the size of the data (data_size), the size of the parameters (theta_size) and the data loader for x and y (data_loader and theta_loader)!")
            
            #Initialize y and x to a empty list (thery are not used in this case)
            self.y = []
            self.x = []
            
            #Take the number of simulations
            self.Ntrain = int(data["Ntrain"])
            if("Nvalidation" in data.keys()):
                self.Nvalidation = int(data["Nvalidation"])
            else:
                self.Nvalidation = 0
                
            #Take the number of channels
            if("Nchannels" in data.keys()):
                Nchannels = data["Nchannels"]
            else:
                Nchannels = 1

            #Take the dimensionality of the data
            Ndim = len(data["data_size"])

            #Raise an error in case the dimension of data is larger than 1 and Posterior was not chosen
            if(Ndim > 1 and kind != "posterior"):
                raise ValueError("The joint and likelihood inferences only work for uni-dimenisonal data!")
            
            #Set the dimensions of data and theta
            if(kind == "joint"):
                self.y_size = data["theta_size"] + data["data_size"][0]
                self.x_size = 0
                self.data_loader_y = lambda inds: jnp.hstack([data["theta_loader"](inds), data["data_loader"](inds)])
                self.data_loader_x = lambda inds: None

            elif(kind == "posterior"):
                self.y_size = data["theta_size"]
                if(Ndim == 1 and Use_conv == False):
                    self.x_size = data["data_size"][0]   
                else:
                    self.x_size = data["data_size"]
                self.data_loader_y = data["theta_loader"]
                self.data_loader_x = data["data_loader"]

            elif(kind == "likelihood"):
                self.y_size = data["data_size"][0] 
                self.x_size = data["theta_size"]
                self.data_loader_y = data["data_loader"]
                self.data_loader_x = data["theta_loader"]

        #Raise an error in case the wrong type was given
        else:
            raise TypeError("Data must be a jax array or a dictionary!")
        
        #Initialize the model
        if(model == None):
            self.models = [CNF(y_size = self.y_size, x_size = self.x_size, key = key, num_blocks = num_blocks, Nneurons = Nneurons, Nneurons_mean_std = Nneurons_mean_std, Nconv = Nconv, Nlin = Nlin, compacted_mult = compacted_mult, momentum_size = momentum_size, Nchannels = Nchannels, compact_type = compact_type)]
        else:
            self.models = [model]

        #Set the best model and losses
        self.models.append(self.models[0])
        self.models.append(self.models[0])
        self.loss_best = [jnp.inf, jnp.inf]
        self.losses = []
        self.losses_validation = []
        self.lr_history = []

        #Save the type of compactfication
        self.compact_type = compact_type
                
    #Get the best model and loss
    def get_best(self):
        if(self.Nvalidation > 0):
            return self.models[1], self.loss_best[0], self.losses, self.models[2], self.loss_best[1], self.losses_validation
        else:
            return self.models[1], self.loss_best[0], self.losses

    #Compute the logP for some data (used to compute in the validation set)
    def LogP_validation(self, Nbatches: Int, key: PRNGKeyArray):
        #Compute the batch size
        batch_size = int(self.Nvalidation/Nbatches)  
        
        #Define the indexes for the validation
        inds = jnp.arange(self.Ntrain, self.Ntrain + self.Nvalidation)

        #Pre-compute the kernels used in the convolutions
        if(self.compact_type == "equivariant"):
            for func in self.models[0].funcs:
                for layer in func.layers:
                    layer.compact_x.compute_kernels()
            for layer in self.models[0].Mean_std.layers_mu:
                layer.compute_kernels()
            for layer in self.models[0].Mean_std.layers_std:
                layer.compute_kernels()

        #Make one step for each batch
        logP = 0.0
        key_loss = jrandom.split(key, Nbatches)
        for i in range(Nbatches):
            y = self.data_loader_y(inds[i*batch_size : (i+1)*batch_size])
            x = self.data_loader_x(inds[i*batch_size : (i+1)*batch_size])
            
            _, logP_batch, _ = jax.vmap(self.models[0].get_logP, in_axes = (0, None, None, 0))(y, key_loss[i], jnp.array([self.models[0].t0]), x)
            logP += jnp.mean(logP_batch)
                
        #Save the loss of this step
        logP = logP/Nbatches
        
        return -logP
    
    #Train the CNF
    def train(self, Nsteps: Int, Nbatches: Int, print_every: Int, optim: optax._src.base.GradientTransformationExtraArgs, key: PRNGKeyArray, optim_state: tuple = None, lr_schedule: optax._src.base.GradientTransformationExtraArgs = None, lr_schedule_state: tuple = None, Ntimes: Int = 1, poly_order: Int = 0, alpha: Float = 0.0, beta: Float = 0.0, weight_params: Float[Array, "data_size"] = None, Shift: bool = False, Rotate: bool = False, suffix: str = None, lr_limit: Float = 1e-4):             
        #Compute the batch size
        batch_size = int(self.Ntrain/Nbatches)

        #Check the dimensions of data if shift or rotation is true
        if((Shift == True or Rotate == True) and len(self.x_size) != 3):
            raise ValueError("The dimension of x must be 3 for applying a rotation and/or a shift!")
        
        #Check if Ntimes is equal to 2
        if(Ntimes == 2):
            raise ValueError("Ntimes must be 1 (no polynomial regularization) or >=3 (polynomial regularization with Ntimes points)!")
        
        #Check if the order o polynomials is larger than 0 in case Ntimes is greater than 1
        if(Ntimes > 1 and poly_order <= 0):
            raise ValueError("The poly_order must be >=1 for the polynomial regularization!")
        
        #Check if alpha is larger than 0
        if(Ntimes > 1 and alpha <=0.0):
            raise ValueError("Alpha must be larger than 0 for the polynomial regularization!")
        
        #Create a folder for the output
        if(suffix is not None):
            try:
                os.system("mkdir Outputs/")
            except:
                pass
        
        #Create the first optim state
        if(optim_state is None):    
            #Split the model between the trainable and fixed parameters
            len_static = len(get_static(self.models[0]))
            filter_spec = jtu.tree_map(lambda _: True, self.models[0])
            filter_spec = eqx.tree_at(get_static, filter_spec, replace = [False for i in range(len_static)])
            diff_model = eqx.filter(pytree = self.models[0], filter_spec = filter_spec) 
            
            optim_state = optim.init(diff_model)

        #Create the learning rate schedule
        if(lr_schedule is None):
            lr_schedule = optax.contrib.reduce_on_plateau(patience = 10, cooldown = 0, factor = 0.5, rtol = 1e-5)

        #Create the first state of the lr_chedule
        if(lr_schedule_state is None):
            #Split the model between the trainable and fixed parameters
            len_static = len(get_static(self.models[0]))
            filter_spec = jtu.tree_map(lambda _: True, self.models[0])
            filter_spec = eqx.tree_at(get_static, filter_spec, replace = [False for i in range(len_static)])
            diff_model = eqx.filter(pytree = self.models[0], filter_spec = filter_spec) 
                        
            lr_schedule_state = lr_schedule.init(diff_model)
        
        #Create the key used in each step to split the data in batches
        key_step, key_validation = jrandom.split(key, 2)
        keys_steps = jrandom.split(key_step, Nsteps)

        #Run the main training loop
        for step, bkey in enumerate(keys_steps):
            
            #Indexes used to shuffle the dataset
            key_shuffle, key_losses, key_shift, key_rotate, key_ts = jrandom.split(bkey, 5)
            inds = jrandom.permutation(key_shuffle, jnp.arange(self.Ntrain))

            #Create the array used to shift the grid
            if(Shift == True):
                shift = jrandom.randint(key_shift, (self.Ntrain, 3), minval = jnp.array([0, 0, 0]), maxval = jnp.array(self.x_size))
            else:
                shift = []

            #Create the array used to rotate the grid
            if(Rotate == True):
                rotation = jrandom.randint(key_rotate, (self.Ntrain, 3), minval = jnp.array([0, 0, 0]), maxval = jnp.array([4, 4, 4]))
            else:
                rotation = []

            #Create the array of times for the polynomial regularization
            if(Ntimes == 1):
                save_ts = jnp.array([self.models[0].t0])
                MatrixT = None
            else:
                save_ts = jnp.sort(jrandom.uniform(key_ts, (Ntimes - 2,), minval = self.models[0].t0, maxval = self.models[0].t1))[::-1]
                save_ts = jnp.hstack([self.models[0].t1, save_ts, self.models[0].t0])
                MatrixT = jnp.array([save_ts**i for i in range(poly_order + 1)]).T
                MatrixT = jnp.identity(Ntimes) - jnp.matmul(jnp.matmul(MatrixT, jnp.linalg.inv(jnp.matmul(jnp.transpose(MatrixT), MatrixT))), jnp.transpose(MatrixT))

            #Compute the loss for the validation
            if(self.Nvalidation > 0):
                self.losses_validation.append(self.LogP_validation(Nbatches = Nbatches, key = key_validation))
                lr_loss = self.losses_validation[-1]
                
                #Save the best model so far in the validation
                if(self.losses_validation[-1] < self.loss_best[1]):
                    self.models[2] = self.models[0]
                    self.loss_best[1] = self.losses_validation[-1]

            #Make one step for each batch
            loss_step = 0.0
            key_loss = jrandom.split(key_losses, Nbatches)
            for i in range(Nbatches):                    
                #Make update the network weigts using this batch
                self.models[0], loss_batch, optim_state = make_step(loss, self.models[0], self.data_loader_y(inds[i*batch_size : (i+1)*batch_size]), optim, key_loss[i], save_ts, MatrixT, alpha, optim_state, lr_schedule_state, self.data_loader_x(inds[i*batch_size : (i+1)*batch_size], shift[i*batch_size : (i+1)*batch_size], rotation[i*batch_size : (i+1)*batch_size]), self.compact_type, beta, weight_params)
                
                #Save the loss of this batch
                loss_step += loss_batch
                
            #Save the loss of this step
            (self.losses).append(loss_step/Nbatches)
            
            #Save the best model so far in the training
            if(self.losses[-1] < self.loss_best[0]):
                self.models[1] = self.models[0]
                self.loss_best[0] = self.losses[-1]

            #Adjusts the learning rate scaling value 
            len_static = len(get_static(self.models[0]))
            filter_spec = jtu.tree_map(lambda _: True, self.models[0])
            filter_spec = eqx.tree_at(get_static, filter_spec, replace = [False for i in range(len_static)])
            diff_model = eqx.filter(pytree = self.models[0], filter_spec = filter_spec) 
            _, lr_schedule_state = lr_schedule.update(updates = diff_model, state = lr_schedule_state, value = lr_loss)
            self.lr_history.append(lr_schedule_state.scale)
                
            #Print partial results and save the models 
            if(step % print_every == 0):
                if(type(suffix) == str):
                    try:
                        os.system("rm Outputs/Losses_%s.h5" %(suffix))
                    except:
                        pass
                
                if(self.Nvalidation > 0):
                    print("Step = %d, Loss_training = %.4f, Loss_validation = %.4f" %(step, self.losses[-1], self.losses_validation[-1]))
                    
                    if(type(suffix) == str):
                        #Save models (only the parameters fitted)
                        len_static = len(get_static(self.models[1]))
                        filter_spec = jtu.tree_map(lambda _: True, self.models[1])
                        filter_spec = eqx.tree_at(get_static, filter_spec, replace = [False for i in range(len_static)])
                        diff_model = eqx.filter(self.models[1], filter_spec)   
                        eqx.tree_serialise_leaves("Outputs/Model_training_%s.eqx" %(suffix), diff_model)

                        len_static = len(get_static(self.models[2]))
                        filter_spec = jtu.tree_map(lambda _: True, self.models[2])
                        filter_spec = eqx.tree_at(get_static, filter_spec, replace = [False for i in range(len_static)])
                        diff_model = eqx.filter(self.models[2], filter_spec)   
                        eqx.tree_serialise_leaves("Outputs/Model_validation_%s.eqx" %(suffix), diff_model)

                        #Save losses
                        f = h5.File("Outputs/Losses_%s.h5" %(suffix), "w")
                        f.create_dataset("loss_training", data = self.losses)
                        f.create_dataset("loss_validation", data = self.losses_validation)
                        f.create_dataset("lr_history", data = self.lr_history)
                        f.close()
                        
                else:
                    print("Step = %d, Loss_training = %.4f" %(step, self.losses[-1]))
                    
                    if(type(suffix) == str):
                        #Save model (only the parameters fitted)
                        len_static = len(get_static(self.models[1]))
                        filter_spec = jtu.tree_map(lambda _: True, self.models[1])
                        filter_spec = eqx.tree_at(get_static, filter_spec, replace = [False for i in range(len_static)])
                        diff_model = eqx.filter(self.models[1], filter_spec)   
                        eqx.tree_serialise_leaves("Outputs/Model_training_%s.eqx" %(suffix), diff_model)   

                        #Save losses
                        f = h5.File("Outputs/Losses_%s.h5" %(suffix), "w")
                        f.create_dataset("loss_training", data = self.losses)
                        f.create_dataset("lr_history", data = self.lr_history)
                        f.close()
                    
            #Stop the loop if the learning rate got very small
            if(lr_schedule_state.scale <= lr_limit):
                print("The CNF converged!")
                break
                    
        #Return the state
        return optim_state

#Select some of the sampled parameters using a simple ABC
def ABC(x, data0, Nparam = 2, eps = 1.0):
    data = x[: , Nparam:]
    params = x[:, :Nparam]
    std2 = jnp.var(data, axis = 0)
    
    #Compute the differences
    diffs = jnp.mean(jnp.power(data - data0, 2.0)/std2, axis = 1)
    mask = diffs < eps
        
    return params[mask, :]

#Test the posterior using the SBC (simulation based calibration)
def SBC_posterior(model: CNF, data: Union[Float[Array, "Nsims Ndim_data"], dict], key: PRNGKeyArray, Nsamples: Int = 1000, theta: Float[Array, "Nsims Ndim_param"] = None,  prior: Callable = lambda _: 1, max_prior: Float = 0.0):
    #Set the parameters of the data and theta in case the data was given as an array
    if(type(data) == jaxlib.xla_extension.ArrayImpl):
        #Check if the data and parameter vector were given to compute the conditional PDFs
        if(theta == None):
            raise ValueError("You have to give a theta array!\n")
      
        #Check if the first dimension of theta and data is equal
        if(theta.shape[0] != data.shape[0]):
            raise ValueError("The first dimensions of theta and data must be the same! %d != %d\n" %(theta.shape[0], data.shape[0]))
    
