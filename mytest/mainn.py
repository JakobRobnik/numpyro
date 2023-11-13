import jax
import jax.numpy as jnp

import os
# num_cores = 128 #specific to perlmutter
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)
num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

import numpy as np
from mytest.benchmarks import *
from mytest.infer import mcmc



def find_crossing(array, cutoff):
    """the smallest M such that array[m] < cutoff for all m > M"""

    def step(carry, element):
        """carry = (, 1 if (array[i] > cutoff for all i < current index) else 0"""
        above_threshold = element > cutoff
        never_been_below = carry[1] * above_threshold  #1 if (array[i] > cutoff for all i < current index) else 0
        return (carry[0] + never_been_below, never_been_below), above_threshold

    state, track = jax.lax.scan(step, init=(0, 1), xs=array, length=len(array))

    return state[0]
    #return jnp.sum(track) #total number of indices for which array[m] < cutoff



def ess(mchmc, Target, num_warmup, num_samples, a, L):

        
    thinning = 1
    
    # run the chains
    samples, _steps = mcmc(mchmc, Target.target, Target.prior, 
                           num_warmup= num_warmup, num_samples= num_samples, thinning= thinning, num_chains= num_cores, 
                           a= a, L= L, progress_bar= True)
    steps = jnp.cumsum(_steps, axis = 1)

    x= Target.tovec(samples)
    print(x.shape)
    # bias of the second moments
    obs = jnp.square(x)
    m = jnp.cumsum(obs, axis = 1) / jnp.arange(1, 1 + num_samples // thinning)[None, :, None]
    
    bias_d = jnp.square(m - Target.m2[None, None, :]) / Target.varm2[None, None, :]
    bias_max = jnp.average(bias_d, axis = -1)
    print(bias_max.shape)
    neff = 100.
    b2_required = 1./neff
    
    # if jnp.any(bias_max[:,-1] > b2_required):
    #     print("Warning: Some chains did not achieve the desire accuracy on " + str(Target.name))
    #     #return 0.
    B2 = jnp.median(bias_max, axis = 0)
    n = find_crossing(B2, b2_required)
    grads = jnp.median(steps[:, n], axis = 0)

    return neff / grads


    

def nuts():
    
            # target,  burn-in, samples, thinning
    setup = [(STN(), 500, 10000), 
             (ICG(), 500, 20000), 
             (Banana(), 500, 20000), 
             (Rosenbrock(), 500, 20000),
             (Neal(), 1000, 100000),
             (Funnel(), 500, 30000), 
            ]
             #(StochasticVolatility(), 500, 1000, 1)]
    
    A = [0.6, 0.65, 0.7, 0.75, 0.8]
    
    for mchmc in [False, True]:
        for itarget in range(len(setup)):
            data = [ess(mchmc, *setup[itarget], a, None) for a in A]
    
            np.save('data/' + setup[itarget][0].name  + '_'+ ('MCHMC' if mchmc else 'HMC')+ '.npy', data)




def hmc():
    
            # target,  burn-in, samples, thinning
    setup = [(STN(), 500, 10000), 
             (ICG(), 500, 20000), 
             (Banana(), 500, 20000), 
             (Rosenbrock(), 500, 20000),
             (Neal(), 1000, 100000),
             (Funnel(), 500, 30000), 
            ]
             #(StochasticVolatility(), 500, 1000, 1)]
    
    Larr = [1, 2, 3, 4, 5, 6]

    ess(False, *setup[0], 0.75, 3)
    
    return 
    
    for mchmc in [False, ]:
        for itarget in range(1):#len(setup)):
            name = setup[itarget][0].name
            print(name, mchmc)
            data = np.array([ess(mchmc, *setup[itarget], 0.8, L) for L in Larr])
            np.save('mytest/data/ess/hmc/' + ('MCHMC' if mchmc else 'HMC') +'/' + name + '.npy', data)



hmc()
