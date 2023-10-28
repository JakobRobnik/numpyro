from tst import run
from benchmarks import *

import matplotlib.pyplot as plt
import numpy as np
plt.style.use(['seaborn-v0_8-talk'])


# example usage

Target = Funnel()

for mchmc in [False, True]:
    method = 'MCHMC' if mchmc else 'HMC'
    print('Running ' + method)
    samples, steps = run(mchmc = mchmc, target= Target.target, prior= Target.prior, 
                num_warmup = 500, num_samples = 10000, num_chains = 1, thinning = 1, 
                progress_bar = True, a= 0.8)

    plt.hist(samples['theta'], bins = 40, density= True, alpha= 0.5, color = ['teal' if mchmc else 'black'], label = 'NUTS '+ method)

theta_true = Target.truth['theta']
plt.plot([theta_true, theta_true], [0., 1.], color = 'tab:red', label = 'true value')

plt.ylabel('marginal posterior density')
plt.xlabel(r'$\vartheta$')
plt.legend()
plt.show()