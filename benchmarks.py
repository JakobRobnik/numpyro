import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import SP500, load_dataset

import numpy as np
from scipy.stats import special_ortho_group




# util

def load_moments(name):
    m2 = jnp.load('ground_truth/' + name + '_m2.npy')
    varm2 = jnp.load('ground_truth/' + name + '_varm2.npy')
    
    return m2, varm2

def neal_prior(d):
    def rng(key):
        key1, key2 = jax.random.split(key)
        theta = numpyro.sample("theta", dist.Normal(0, 3), rng_key=key1)
        z = numpyro.sample("z", dist.Normal(jnp.zeros(d - 1), jnp.exp(0.5 * theta)), rng_key= key2)
        return {'theta': theta, 'z': z}
    return rng


def x_to_vec(y):
    return jnp.array(y['x'])

def vec_to_x(x):
    return {'x': x}



# Ill condition number and bimodality
    

class ICG():
    """Inference gym's Ill conditioned Gaussian"""

    def __init__(self):
        
        self.name = 'icg'
        self.tovec = x_to_vec
        
        d = 100
        self.d = d
        
        rng = np.random.RandomState(seed=10 & (2 ** 32 - 1))
        eigs = np.sort(rng.gamma(shape=0.5, scale=1., size= d)) #eigenvalues of the Hessian
        eigs *= jnp.average(1.0/eigs)
        self.eigs = eigs
        R, _ = np.linalg.qr(rng.randn(d, d)) #random rotation
        self.Sigma = R @ np.diag(1/eigs) @ R.T
        
        self.m2 = jnp.diagonal(R @ np.diag(1.0/eigs) @ R.T)
        self.varm2 = 2 * jnp.square(self.m2)
            
    def target(self):
        numpyro.sample('x', dist.MultivariateNormal(jnp.zeros(self.d), self.Sigma))
        
    def prior(self, key):
        return vec_to_x(numpyro.sample('x', dist.Normal(jnp.zeros(self.d), jnp.ones(self.d) * jnp.max(1.0/jnp.sqrt(self.eigs))), rng_key = key))



class Bimodal():
    """Bi-modal Gaussian"""
    def __init__(self):
        self.name = 'bimodal'
        self.tovec = x_to_vec
        
        d = 50
        self.mu1 = jnp.zeros(d)
        a, f = 7., 0.2
        self.mu2 = jnp.concatenate((jnp.ones(1) * a, np.zeros(d-1)))
        self.f = f
        
        self.m2 = jnp.concatenate((jnp.ones(1) * (1 + f*a**2), np.zeros(d-1)))
        self.varm2 = jnp.concatenate((jnp.ones(1) * (2 + 4*f*a**2 + f*(1-f)*a**4), 2 * np.zeros(d-1)))
        
        self.prior = lambda key: vec_to_x(numpyro.sample('x', dist.Normal(jnp.zeros(d), jnp.ones(d)), rng_key = key))
        
                
    def target(self):
        mix = dist.Categorical(jnp.array([(1-self.f), self.f]))
        component_dist = dist.Normal(loc=np.array([self.mu1, self.mu2]).T)
        mixture = dist.MixtureSameFamily(mix, component_dist)
        numpyro.sample('x', mixture)



# bananas

class Banana():
    """2 d Banana from the inference gym"""
    
    def __init__(self):
        self.name = 'banana'
        self.m2, self.varm2 = load_moments(self.name)
        self.tovec = lambda y: jnp.stack([y['x'], y['y']], axis = -1)  
        
        self.curvature = 0.03

    def target(self):
        x = numpyro.sample("x", dist.Normal(0, 10.0))
        numpyro.sample("y", dist.Normal(self.curvature * (jnp.square(x) - 100.0), 1.0))

    def prior(self, key):
        key1, key2 = jax.random.split(key)
        return {'x': numpyro.sample('x', dist.Normal(), rng_key = key1),
                'y': numpyro.sample('x', dist.Normal(), rng_key = key2)}
    
    def exact_samples(self, key):
        key1, key2 = jax.random.split(key)
        x = numpyro.sample("x", dist.Normal(0, 10.0), rng_key = key1)
        y = numpyro.sample("y", dist.Normal(self.curvature * (jnp.square(x) - 100.0), 1.0), rng_key = key2)
        return jnp.array([x, y])



class Rosenbrock():
    """see https://en.wikipedia.org/wiki/Rosenbrock_function (here we set a = 1, b = 1/Q)"""

    def __init__(self):
        
        self.name = 'rosenbrock'
        self.m2, self.varm2 = load_moments(self.name)
        self.tovec = lambda y: jnp.concatenate((y['x'], y['y']), axis = -1)
        
        self.Q = 0.1
        self.ones = jnp.ones(18)
    
        
    def target(self):
        x = numpyro.sample("x", dist.Normal(self.ones, self.ones))
        numpyro.sample("y", dist.Normal(jnp.square(x), np.sqrt(self.Q) * self.ones))

    def prior(self, key):
        key1, key2 = jax.random.split(key)
        return {'x': numpyro.sample('x', dist.Normal(self.ones, self.ones), rng_key = key1),
                'y': numpyro.sample('x', dist.Normal(self.ones, self.ones), rng_key = key2)}



    def exact_samples(self, key):
        key1, key2 = jax.random.split(key)
        x = numpyro.sample("x", dist.Normal(self.ones, self.ones), rng_key = key1)
        y = numpyro.sample("y", dist.Normal(jnp.square(x), np.sqrt(self.Q) * self.ones), rng_key = key2)
        return jnp.concatenate((x, y))



# hierarchical models

class Neal():
    """Neal's funnel"""
    
    def __init__(self):

        self.name = 'neal'
        self.m2, self.varm2 = load_moments(self.name)
        self.tovec = lambda y: jnp.concatenate((jnp.expand_dims(y['theta'], -1), y['z']), axis = -1)
        self.d = 20
        self.prior = neal_prior(self.d)

    def target(self):
        theta = numpyro.sample("theta", dist.Normal(0, 3))
        numpyro.sample("z", dist.Normal(jnp.zeros(self.d - 1), jnp.exp(0.5 * theta)) )

    def exact_samples(self, key):
        key1, key2 = jax.random.split(key)
        theta = numpyro.sample("theta", dist.Normal(0, 3), rng_key = key1)
        z = numpyro.sample("z", dist.Normal(jnp.zeros(self.d - 1), jnp.exp(0.5 * theta)), rng_key = key2)
        return jnp.concatenate((theta * jnp.ones(1), z))


class Funnel():
    """Funnel with noisy data"""
    
    def __init__(self):
        
        self.name = 'funnel'
        self.m2, self.varm2 = load_moments(self.name)
        self.tovec = lambda y: jnp.concatenate((jnp.expand_dims(y['theta'], -1), y['z']), axis = -1)
                
        self.d = 20
        self.sigma = 1.
        
        self.prior = neal_prior(self.d)
        
        # generate the data
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        self.truth = self.prior(key1)
        self.data = numpyro.sample("zobs", dist.Normal(self.truth['z'], self.sigma * jnp.ones(self.d-1)), rng_key= key2)
        
    def target(self):
        theta = numpyro.sample("theta", dist.Normal(0, 3))
        z = numpyro.sample("z", dist.Normal(jnp.zeros(self.d - 1), jnp.exp(0.5 * theta)) )
        numpyro.sample("zobs", dist.Normal(z, self.sigma), obs = self.data)



class StochasticVolatility():
    
    def __init__(self):
        
        self.name= 'sv'
        self.m2, self.varm2 = None, None# load_moments(self.name)
        
        self.tovec = lambda y: jnp.concatenate((jnp.expand_dims(y['sigma'], -1), jnp.expand_dims(y['nu'], -1), y['s']), axis = -1)
        
        _, fetch = load_dataset(SP500, shuffle=False)
        SP500_dates, SP500_returns = fetch()
        self.data = SP500_returns
        

    def target(self):
        sigma = numpyro.sample("sigma", dist.Exponential(50.0))
        nu = numpyro.sample("nu", dist.Exponential(0.1))
        s = numpyro.sample("s", dist.GaussianRandomWalk(scale=sigma, num_steps=jnp.shape(self.data)[0]))
        numpyro.sample("r", dist.StudentT(df=nu, loc=0.0, scale=jnp.exp(s)), obs= self.data)

    
    def prior(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        sigma = numpyro.sample("sigma", dist.Exponential(50.0), rng_key= key1)
        nu = numpyro.sample("nu", dist.Exponential(0.1), rng_key= key2)
        s = numpyro.sample("s", dist.GaussianRandomWalk(scale=sigma, num_steps=jnp.shape(self.data)[0]), rng_key= key3)
        return {'sigma': sigma, 'nu': nu, 's': s}


