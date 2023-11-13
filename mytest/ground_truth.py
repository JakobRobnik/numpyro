import jax
import jax.numpy as jnp
import os

num_cores = 256 #specific to perlmutter
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)
num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

from mytest.benchmarks import *
from mytest.infer import mcmc


key = jax.random.PRNGKey(42)



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

    samples, _ = mcmc(False, Target.target, Target.prior, 
                      num_warmup= num_warmup, num_samples= num_samples, thinning= thinning, num_chains= num_cores, 
                      progress_bar= True)
    
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
    
    
    

run_ground_truth()