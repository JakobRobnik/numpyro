import jax
import jax.numpy as jnp
from numpyro.infer import HMC, NUTS, MCMC



def mcmc(mchmc, target, prior, 
        num_warmup= 500, num_samples= 3000, num_chains= 1, thinning= 1, 
        a= 0.8, L = None,
        progress_bar = False, random_key = None):
    
    """runs MCMC chains with either HMC or MCHMC
        Args: 
            mchmc: will run MCHMC if True
            a: acceptance rate
            L: number of leapfrog steps per sample. If None we run NUTS.
    """
    
    if random_key == None:
        key = jax.random.PRNGKey(42)
    else:
        key = random_key

    key_sampling, key_warmup, key_prior = jax.random.split(key, 3)
    if L == None:
        kernel = NUTS(target, adapt_step_size=True, adapt_mass_matrix= False, dense_mass=False, mchmc= mchmc, target_accept_prob= a)
    else:
        kernel = HMC(target, num_steps = L, adapt_step_size= False, adapt_mass_matrix= False, dense_mass=False, mchmc= mchmc, target_accept_prob= a, step_size= 0.4)
        
    sampler = MCMC(kernel, num_warmup= num_warmup, num_samples= num_samples, num_chains= num_chains, thinning= thinning, progress_bar= progress_bar)

    if num_chains > 1:
        x0 = jax.vmap(prior)(jax.random.split(key_prior, num_chains))
    else:
        x0 = prior(key_prior)
        
    sampler.warmup(key_warmup, init_params= x0, extra_fields=['num_steps'], collect_warmup=True)
    sampler.run(key_sampling, extra_fields=['num_steps'])
    
    steps = jnp.array(sampler.get_extra_fields(group_by_chain= True)['num_steps'], dtype=int)

    return sampler.get_samples(group_by_chain= True), steps
