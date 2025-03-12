import types

import blackjax
import jax
import jax.numpy as jnp
import tqdm


jax.config.update('jax_enable_x64', True) 
jax.config.update('jax_platform_name', 'cpu')

rng_key = jax.random.PRNGKey(0)
n_dims = 3
n_live = 1000
n_delete = 500
num_mcmc_steps = n_dims * 5

# | Define data and likelihood
x = jnp.linspace(-1, 1, 10)
m = 2.0
c = 1.0
sigma = 0.1
key, rng_key = jax.random.split(rng_key)
y =  m * x + c + sigma * jax.random.normal(key, (10,), dtype=jnp.float64)

#plt.errorbar(x, y, yerr=sigma, fmt="o")
#plt.plot(x, m * x + c)

@jax.jit
def loglikelihood_fn(p):
    return jax.scipy.stats.multivariate_normal.logpdf(y, p["m"] * x + p["c"], p["sigma"])

# | Define the prior function

m_min, m_max = -10.0, 10.0
c_min, c_max = -10.0, 10.0
sigma_min, sigma_max = 0.0, 10.0

uniform_logprob = lambda x, a, b: jax.scipy.stats.uniform.logpdf(x, a, b-a)

def logprior_fn(p):
    logprior = 0.0
    logprior += uniform_logprob(p["m"], m_min, m_max)
    logprior += uniform_logprob(p["c"], c_min, c_max)
    logprior += uniform_logprob(p["sigma"], sigma_min, sigma_max)
    return logprior

# | Sample live points from the prior
rng_key, init_key = jax.random.split(rng_key, 2)
init_keys = jax.random.split(init_key, n_dims)
particles = {
        "m": jax.random.uniform(init_keys[0], (n_live,), minval=m_min, maxval=m_max),
        "c": jax.random.uniform(init_keys[1], (n_live,), minval=c_min, maxval=c_max),
        "sigma": jax.random.uniform(init_keys[2], (n_live,), minval=sigma_min, maxval=sigma_max),
        }

_, ravel_fn = jax.flatten_util.ravel_pytree({k: v[0] for k, v in particles.items()})

# | Initialize the Nested Sampling algorithm
nested_sampler = blackjax.ns.adaptive.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
    ravel_fn=ravel_fn,
)

state = nested_sampler.init(particles, loglikelihood_fn)

@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = nested_sampler.step(subk, state)
    return (state, k), dead_point

# | Run Nested Sampling
dead = []
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(n_delete)  # Update progress bar


# replace by utils: finalise from util.py to zip NestedInfo together
# | anesthetic post-processing
from anesthetic import NestedSamples
import numpy as np
dead = jax.tree.map(
        lambda *args: jnp.reshape(jnp.stack(args, axis=0), 
                                  (-1,) + args[0].shape[1:]),
        *dead)

live = state.sampler_state
logL = np.concatenate((dead.logL, live.logL), dtype=float)
logL_birth = np.concatenate((dead.logL_birth, live.logL_birth), dtype=float)
pid = np.concatenate((dead.pid, live.pid), dtype=int)
start_pid = np.concatenate((dead.mcmc_chain_start_pid, -1 * jnp.ones(n_live, dtype=int)), dtype=int)
end_pid = np.concatenate((dead.mcmc_chain_end_pid, -1 * jnp.ones(n_live, dtype=int)), dtype=int)
data = np.concatenate([
    np.column_stack([v for v in dead.particles.values()]),
    np.column_stack([v for v in live.particles.values()])
    ], axis=0)

columns = list(dead.particles.keys())
samples = NestedSamples(data, logL=logL, logL_birth=logL_birth, columns=columns)
samples['pid'] = pid
samples['start_pid'] = start_pid
samples['end_pid'] = end_pid
samples.gui()
samples.to_csv('line.csv') 


# BEGINN:
# starting point are the current compression factors t_i
print('compression factors')
t = np.log(samples.nlive/(samples.nlive+1))
print(t)
print('t.shape', t.shape)
# or the logX
logX = samples.logX()

# first compression happens at volume of last particle deleted in the first iteration
log_t1 = logX.iloc[n_delete] - logX.iloc[0]
print('t1', np.exp(log_t1))
print('log_X0', logX.iloc[0])
print('log_X1', logX.iloc[1])
# estimating t1 using mcmc samples with the loglikelihoods of the mcmc samples -- this is likely the one you need
# print('dead mcmc chain loglikelihood shape', dead.mcmc_chain.loglikelihood.shape)

# print(dead.mcmc_chain.loglikelihood.shape)
# Get first n_delete mcmc points to estimate volume
first_iteration_loglikelihoods = jnp.ravel(dead.mcmc_chain.loglikelihood[:n_delete, :])
contours = sorted(list(set([l for l in logL_birth.tolist() if not np.isinf(l)])))
print('logL_births.shape', logL_birth.shape)
print('contours', len(contours), contours)
first_contour = contours[1]  # why shouldn't this be index 0??
print('first_contour', first_contour)
# print('chain log likelihoods', mcmc_chain.loglikelihood[:n_delete][chain_idx])
print('sampled compression', jnp.mean(first_iteration_loglikelihoods<= first_contour), 'estimated', np.exp(log_t1))

print('original evidence', samples.logZ())
corrected_logX = logX
def corrected_logX_fn(self, nsamples=None):
        """Log-Volume.

        The log of the prior volume contained within each iso-likelihood
        contour.

        Parameters
        ----------
        nsamples : int, optional
            - If nsamples is not supplied, calculate mean value
            - If nsamples is integer, draw nsamples from the distribution of
              values inferred by nested sampling

        Returns
        -------
        if nsamples is None:
            WeightedSeries like self
        elif nsamples is int:
            WeightedDataFrame like self, columns range(nsamples)
        """
        if nsamples is None:
            t = np.log(self.nlive / (self.nlive + 1))
        else:
            r = np.log(np.random.rand(len(self), nsamples))
            w = self.get_weights()
            r = self.nlive._constructor_expanddim(r, self.index, weights=w)
            t = r.divide(self.nlive, axis=0)
            t.columns.name = 'samples'
        # > or <
        log_compression_correction = np.log(jnp.mean(first_iteration_loglikelihoods > first_contour) - t[0])
        print('correction',     log_compression_correction)
        # TODO!! Make comparison with nan's robust
        print('first it ll', first_iteration_loglikelihoods)
        assert jnp.isinf(first_iteration_loglikelihoods).sum() + jnp.isnan(first_iteration_loglikelihoods).sum() == 0
        correction_t = np.log(jnp.mean(first_iteration_loglikelihoods > first_contour))
        logX = t.cumsum()
        logX = logX - logX.iloc[n_delete] + (correction_t+logX.iloc[1])
        logX.name = 'logX'
        return logX

# Overwrite the logX method on this instance
samples.logX = types.MethodType(corrected_logX_fn, samples)
print('corrected evidence', samples.logZ())
print('uncertainty', np.sqrt(samples.D_KL()/n_live))
#

# TODO: show volume estimation is improved

exit(0)


# Q: is there a way to do this without losing named tuple access?
logLs = dead.logL
mcmc_chain = dead.mcmc_chain
mcmc_chain_start_pid = dead.mcmc_chain_start_pid
mcmc_chain_end_pid = dead.mcmc_chain_end_pid
logL_birth = dead.logL_birth

chain_idx = 0
print('mcmc positions for chain', chain_idx)
print(len(mcmc_chain.position['c'][chain_idx]))
print(mcmc_chain.position['c'][chain_idx][0])
print(mcmc_chain.position['m'][chain_idx][0])
print(mcmc_chain.position['sigma'][chain_idx][0])
first_chain_position = {
    'c': mcmc_chain.position['c'][chain_idx][0],
    'm': mcmc_chain.position['m'][chain_idx][0],
    'sigma': mcmc_chain.position['sigma'][chain_idx][0],
    }
last_chain_position = {
    'c': mcmc_chain.position['c'][chain_idx][-1],
    'm': mcmc_chain.position['m'][chain_idx][-1],
    'sigma': mcmc_chain.position['sigma'][chain_idx][-1],
}

print('first mcmc pos', first_chain_position)
print('last mcmc pos', last_chain_position)
print('start and end pids for chain ', chain_idx)
print(mcmc_chain_start_pid[chain_idx])
print(mcmc_chain_end_pid[chain_idx])
print(f'real chain {chain_idx} start loc', samples[samples['pid']==mcmc_chain_start_pid[chain_idx]][['c', 'm', 'sigma']].values)
# print('chain start', first_chain_position)
print(f'real chain {chain_idx} end loc', samples[samples['pid']==mcmc_chain_end_pid[chain_idx]][['c', 'm', 'sigma']].values)
# print('chain last', last_chain_position)

real_start = samples[samples['pid']==mcmc_chain_start_pid[chain_idx]][['c', 'm', 'sigma']].values[0]
real_end = samples[samples['pid']==mcmc_chain_end_pid[chain_idx]][['c', 'm', 'sigma']].values[0]
full_c_chain = np.concatenate([np.array([real_start[0]]), mcmc_chain.position['c'][chain_idx]])
print('chain length', len(full_c_chain))
print('new first c', full_c_chain[0], 'new last c', full_c_chain[-1])
print('real first c', real_start[0], 'real last c', real_end[0])

# TODO: complete all 500 chains
# TODO: use all 500 chains to improve volume estimate for t1 (specifically complete the likelihoods)
# first idea for first contour
# Q: what is the likelihood of the first contour? does it have the idx n_delete or not??
# contour_idx = n_delete
# print('contour limit', samples.contour(contour_idx))
# print('num contours', samples.logL.shape)
# first_contour = samples.contour(contour_idx)
# particles in second iteration should know correct logL_birth as contour
print('len(dead)', logL_birth.shape)
# second idea for contour # did not work
# Q: how to get likelihood contour? -> use logL_birth
# second_iteration_logL_birth = logL_birth[n_live:2*n_live]
# third idea: find the particle with id nlive+1 and take its logL_birth as contour


# A: the real chain start is missing from the mcmc chain (which is what we expect from this implementation of ns)


# log likelihoods
# print(logL_birth)
# print(logL)


dead.mcmc_chain.position['m']
dead.mcmc_chain.position['c']
dead.mcmc_chain.position['sigma']


# Q1: why does the next thing not work?
# print('live mcmc chain loglikelihood shape', live.mcmc_chain.loglikelihood.shape)

# If you like, you can tag these onto the array
chain_loglikelihoods = np.concatenate([dead.mcmc_chain.loglikelihood, np.nan * np.ones((n_live, num_mcmc_steps))], axis=0)
# print('chain loglikelihoods shape', chain_loglikelihoods.shape)
samples[[f'logL_{i}' for i in range(num_mcmc_steps)]] = chain_loglikelihoods

# The log of the prior volume contained within each iso - likelihood contour.
print('samples.logX')
samples.logX()
print(samples.logL.shape)

# now we use the MCMC chains to re-estimate the first (probably best?) compression factor
# use a single  corrected compression factor to correct all volumes
# re-estimate logZ
# Q4: what is meant by eq 4?
