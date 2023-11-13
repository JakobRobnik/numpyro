import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import HMC, NUTS, MCMC
import matplotlib.pyplot as plt
plt.style.use(['seaborn-v0_8-talk'])



# random number generator key
key = jax.random.PRNGKey(42)
key_data1, key_data2, key_warmup, key_sampling = jax.random.split(key, 4)


# setup the taget distribution, we will use Funnel with data    
d = 20
sigma = 1.

# true value of the parameters
def neal_prior(key):
    key1, key2 = jax.random.split(key)
    theta = numpyro.sample("theta", dist.Normal(0, 3), rng_key=key1)
    z = numpyro.sample("z", dist.Normal(jnp.zeros(d - 1), jnp.exp(0.5 * theta)), rng_key= key2)
    return {'theta': theta, 'z': z}
        
truth = neal_prior(key_data1)

# observed data
data = numpyro.sample("zobs", dist.Normal(truth['z'], sigma * jnp.ones(d-1)), rng_key= key_data2)
        
# Bayesian model
def target():
    theta = numpyro.sample("theta", dist.Normal(0, 3))
    z = numpyro.sample("z", dist.Normal(jnp.zeros(d - 1), jnp.exp(0.5 * theta)) )
    numpyro.sample("zobs", dist.Normal(z, sigma), obs = data)


# run MCHMC and HMC

for mchmc in [False, True]:
    method = 'MCHMC' if mchmc else 'HMC'
    print('Running ' + method)
    
    ### the syntax is the standard syntax from numpyro, ###
    ### we just add an optional parameter mchmc= True if we want to run MCHMC ###
    kernel = NUTS(target, adapt_step_size=True, adapt_mass_matrix= False, dense_mass=False, mchmc= mchmc)
    sampler = MCMC(kernel, num_warmup = 500, num_samples = 10000, num_chains = 1)
    sampler.warmup(key_warmup)
    sampler.run(key_sampling)
    samples = sampler.get_samples()
    
    plt.hist(samples['theta'], bins = 40, density= True, alpha= 0.5, color = ['teal' if mchmc else 'black'], label = 'NUTS '+ method)

theta_true = truth['theta']
plt.plot([theta_true, theta_true], [0., 0.4], color = 'tab:red', label = 'true value')

plt.ylabel('marginal posterior density')
plt.xlabel(r'$\vartheta$')
plt.legend()
plt.savefig('posterior.png')
plt.close()
#plt.show()
