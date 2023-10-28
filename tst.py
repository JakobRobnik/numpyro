import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_flatten
from jax.flatten_util import ravel_pytree
import os
#num_cores = 6 #specific to my PC
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)
num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)
import numpy as np
from numpyro.infer import HMC, NUTS, MCMC
from benchmarks import *

import matplotlib.pyplot as plt
import numpy as np
plt.style.use(['seaborn-v0_8-talk'])



key = jax.random.PRNGKey(2)


def run(mchmc, target, prior, 
        num_warmup = 500, num_samples = 3000, num_chains = 1, thinning = 1, 
        progress_bar = False, a= 0.8):
    """runs NUTS chains with either HMC or MCHMC"""
    
    key_sampling, key_warmup, key_prior = jax.random.split(key, 3)

    nuts_setup = NUTS(target, adapt_step_size=True, adapt_mass_matrix= False, dense_mass=False, mchmc= mchmc, target_accept_prob= a)

    sampler = MCMC(nuts_setup, num_warmup= num_warmup, num_samples= num_samples, num_chains= num_chains, thinning= thinning, progress_bar= progress_bar)

    if num_chains > 1:
        x0 = jax.vmap(prior)(jax.random.split(key_prior, num_chains))
    else:
        x0 = prior(key_prior)
    sampler.warmup(key_warmup, init_params= x0, extra_fields=['num_steps'], collect_warmup=True)
    sampler.run(key_sampling, extra_fields=['num_steps'])
    
    steps = np.array(sampler.get_extra_fields(group_by_chain= True)['num_steps'], dtype=int)

    return sampler.get_samples(group_by_chain= True), steps



def exact_ground_truth(Target, num_samples):
    
    print(Target.name)

    x = jax.vmap(Target.exact_samples)(jax.random.split(key, num_samples))
        
    obs = jnp.square(x)
    
    # second moments and their variances
    m1 = jnp.average(obs, axis = 0)
    m2 = jnp.average(jnp.square(obs), axis = 0)
    var = m2 - jnp.square(m1)
        
    jnp.save('ground_truth/' + Target.name + '_m2.npy', m1)
    jnp.save('ground_truth/' + Target.name + '_varm2.npy', var)
    

    

def ground_truth(Target, num_warmup, num_samples, thinning):
    """run very long NUTS HMC chains to compute the ground truth moments for the given target"""
    
    print(Target.name)

    samples, _ = run(False, Target.target, Target.prior, num_warmup= num_warmup, num_samples= num_samples, thinning= thinning, num_chains= num_cores, progress_bar= True)
    x= Target.tovec(samples)
    
    obs = jnp.square(x)
    
    # second moments and their variances
    m1 = jnp.average(obs, axis = (0, 1))
    m1_all = jnp.average(obs, axis = 1)
    m2 = jnp.average(jnp.square(obs), axis = (0, 1))
    var = m2 - jnp.square(m1)
        
    jnp.save('ground_truth/' + Target.name + '_m2.npy', m1)
    jnp.save('ground_truth/' + Target.name + '_varm2.npy', var)
    
    # use individual chains to estimate the error
    bias_d = jnp.square(m1_all - m1[None, :]) / var[None, :]
    bias_max = jnp.sqrt(jnp.max(bias_d, axis = -1))
    print('Bias relative to the all-chain average-> worst chain: {0}, average chain: {1}.'.format(jnp.max(bias_max), jnp.median(bias_max)))



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

    
def ess(mchmc, Target, num_warmup, num_samples, a):
    """computes the ground truth moments for the given target"""
    print(('MCHMC' if mchmc else 'HMC') + ' (a = ' +str(a) +'): ' + Target.name)
        
    thinning = 1
    # run the chains
    samples, _steps = run(mchmc, Target.target, Target.prior, num_warmup= num_warmup, num_samples= num_samples, thinning= thinning, num_chains= num_cores, progress_bar= False, a = a)
    steps = jnp.cumsum(_steps, axis = 1)
    x= Target.tovec(samples)
    
    # bias of the second moments
    obs = jnp.square(x)
    m = jnp.cumsum(obs, axis = 1) / jnp.arange(1, 1 + num_samples // thinning)[None, :, None]
    
    bias_d = jnp.square(m - Target.m2[None, None, :]) / Target.varm2[None, None, :]
    bias_max = jnp.max(bias_d, axis = -1)
    
    #for i in range(len(steps)):
    #     plt.plot(steps[i], bias_max[i], color = 'black', alpha = 0.5)
    # plt.yscale('log')
    # plt.show()
    # smoothen the results by using results from multiple chains
    
    neff = 100.
    b2_required = 1./neff
    
    if jnp.any(bias_max[:,-1] > b2_required):
        print("Warning: Some chains did not achieve the desire accuracy on " + str(Target.name))
        #return 0.
    
    #else:    
    n = jax.vmap(find_crossing, (0, None))(bias_max, b2_required)
    grads = steps[jnp.arange(len(steps)), n] / neff
    return jnp.median(grads)



def run_ground_truth():
    """parameters of the ground truth computation"""
    
            # target,  burn-in, samples, thinning
    setup = [(Banana(), 10**8), 
             (Rosenbrock(), 10**7),
             (Neal(), 10**7),
             (Funnel(), 10000, 10**6, 10), 
             (StochasticVolatility(), 5000, 50000, 10)]
        
    for i in range(2, 3):
        exact_ground_truth(*setup[i])
    
    

def run_ess():
    """parameters of the ess computation"""
    
            # target,  burn-in, samples, thinning
    setup = [(ICG(), 500, 20000), 
             (Banana(), 500, 20000), 
             (Rosenbrock(), 500, 20000),
             (Neal(), 1000, 100000),
             (Funnel(), 500, 30000), 
            ]
             #(StochasticVolatility(), 500, 1000, 1)]
    
    A = [0.6, 0.65, 0.7, 0.75, 0.8]
    which = [3, ]
    #print([[ess(mchmc, *setup[i], 0.8) for i in which] for mchmc in [False, True]])
    
    results_new = [[[ess(mchmc, *setup[i], a) for a in A] for i in which] for mchmc in [False, True]]
    
    results = np.load('ess.npy')
    results[:, which, :] = results_new
    np.save('ess.npy', results)
    
    
    
def gridplot():
    
    data = np.load('ess.npy')
    names = ['ICG', 'Banana', 'Rosenbrock', 'Neal', 'Funnel']
    A = np.array([0.6, 0.65, 0.7, 0.75, 0.8])
    
    colors = ['black', 'tab:blue']
    labels = ['HMC', 'MCHMC']
    
    plt.figure(figsize = (20, 4))
    for itarget in range(len(names)):
        plt.subplot(1, 5, itarget + 1)
        plt.title(names[itarget])
        
        for imethod in range(2):
            x = data[imethod, itarget]
            mask = x > 1.
            plt.plot(A[mask], x[mask], 'o:', color = colors[imethod], label = labels[imethod])
            
        plt.xlabel('acceptance rate')
        if itarget == 0:
            plt.ylabel('# gradients / ESS')
    plt.legend()
    plt.show()
    


