import types
import matplotlib.pyplot as plt
import arviz as az

import blackjax
import blackjax.ns.utils as nsutils
from diagnostics import nested_rhat, lower_bound_ess,
import jax
import jax.numpy as jnp
import tqdm


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
    # import numpy
    # numpy.seterr(all='raise')
    if nsamples is None:
        t = np.log(self.nlive / (self.nlive + 1))
    else:
        r = np.log(np.random.rand(len(self), nsamples))
        w = self.get_weights()
        r = self.nlive._constructor_expanddim(r, self.index, weights=w)
        t = r.divide(self.nlive, axis=0)
        t.columns.name = 'samples'
    # > or <
    assert jnp.isinf(first_iteration_loglikelihoods).sum() + jnp.isnan(first_iteration_loglikelihoods).sum() == 0
    correction_t = np.log(jnp.mean(first_iteration_loglikelihoods > first_contour))
    print('mcmc_ratio', jnp.mean(first_iteration_loglikelihoods > first_contour))
    logX = t.cumsum()
    skilling_t = np.cumsum(t[:n_delete]).iloc[-1]
    print(np.exp(skilling_t))
    print('rescaling logX by', correction_t / skilling_t)
    logX = logX * (correction_t / skilling_t)
    logX.name = 'logX'
    return logX


jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

rng_key = jax.random.PRNGKey(0)
n_dims = 3
factor = 1
n_live = int(factor * 1000)
n_delete = int(factor * 500)
num_mcmc_steps = n_dims * 60

# | Define data and likelihoo
n_data_points = 10
x = jnp.linspace(-1, 1, n_data_points)
m = 2.0
c = 1.0
sigma = 0.1
key, rng_key = jax.random.split(rng_key)
y = m * x + c + sigma * jax.random.normal(key, (n_data_points,), dtype=jnp.float64)


#########
# BEGIN problem
#########
@jax.jit
def loglikelihood_fn(p):
    return jax.scipy.stats.multivariate_normal.logpdf(y, p["m"] * x + p["c"], p["sigma"])

# | Define the prior function
m_min, m_max = -10.0, 10.0
c_min, c_max = -10.0, 10.0
sigma_min, sigma_max = 0.0, 10.0

uniform_logprob = lambda x, a, b: jax.scipy.stats.uniform.logpdf(x, a, b - a)


def logprior_fn(p):
    logprior = 0.0
    logprior += uniform_logprob(p["m"], m_min, m_max)
    logprior += uniform_logprob(p["c"], c_min, c_max)
    logprior += uniform_logprob(p["sigma"], sigma_min, sigma_max)
    return logprior


#########
# END problem
#########

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

blackjax.diagnostics.effective_sample_size()

# _dead = nsutils.finalise(state, dead)
# print('blackjax logZ', jnp.mean(nsutils.logZ(rng_key, _dead, samples=int(1e3))))

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
# start_pid = np.concatenate((dead.mcmc_chain_start_pid, -1 * jnp.ones(n_live, dtype=int)), dtype=int)
# end_pid = np.concatenate((dead.mcmc_chain_end_pid, -1 * jnp.ones(n_live, dtype=int)), dtype=int)
data = np.concatenate([
    np.column_stack([v for v in dead.particles.values()]),
    np.column_stack([v for v in live.particles.values()])
], axis=0)

columns = list(dead.particles.keys())
samples = NestedSamples(data, logL=logL, logL_birth=logL_birth, columns=columns)

# BEGIN:
# starting point are the current compression factors t_i
# print('compression factors')
t = np.log(samples.nlive / (samples.nlive + 1))

skilling_estimate = jnp.sum(t[:n_delete].values)
skilling_error = jnp.sqrt(skilling_estimate)
print('skilling', skilling_estimate, skilling_error)
# print(t)
# print('t.shape', t.shape)
# or the logX
logX = samples.logX()

# first compression happens at volume of last particle deleted in the first iteration
log_t1 = logX.iloc[n_delete] - logX.iloc[0]
print('t1', np.exp(log_t1))
print('log_X0', logX.iloc[0])
print('log_X1', logX.iloc[1])
# estimating t1 using mcmc samples with the loglikelihoods of the mcmc samples -- this is likely the one you need
numpy_samples = {
    k: np.array(v[:n_delete, :]) for k, v in dead.mcmc_chain.position.items()
}

K = 8
nested_rhat_values = {
    k: nested_rhat(numpy_samples[k], K) for k, v in numpy_samples.items()
}
print('nested_rhat_values')
print(nested_rhat_values)

lower_bound_ess_single = lower_bound_ess(nested_rhat_values, numpy_samples['c'].shape[0]/K, numpy_samples['c'].shape[1])
print('lower bound ess per chain', lower_bound_ess_single)
lower_bound_ess_all = {k: numpy_samples['c'].shape[0] * lower_bound_ess_single[k] for k in lower_bound_ess_single.keys()}
print('lower bound ess all chains', lower_bound_ess_all)

print(numpy_samples['c'].shape)
idata = az.from_dict(numpy_samples)
print('arviz ess', az.ess(idata, var_names=['c', 'm', 'sigma']))
print('arviz rel ess', az.ess(idata, relative=True, var_names=['c', 'm', 'sigma']))
print('msce (arviz)', az.mcse(idata))
az.plot_mcse(idata)
plt.show()

# K = 10  # For instance, partition the chains into 10 superchains.
# print(samples)
# c = samples[['c']].values
# m = samples[['m']].values
# sigma = samples[['sigma']].values
# print(c)
# nested_rhat = nested_rhat(c , K)
# print('nested rhat', nested_rhat)

exit(0)

# print(dead.mcmc_chain.loglikelihood.shape)
# Get first n_delete mcmc points to estimate volume
first_iteration_loglikelihoods = jnp.ravel(dead.mcmc_chain.loglikelihood[:n_delete, :])
contours = sorted(list(set([l for l in logL_birth.tolist() if not np.isinf(l)])))
# print('logL_births.shape', logL_birth.shape)
print('contours', len(contours), contours)
first_contour = contours[1]  # why shouldn't this be index 0??
print('first_contour', first_contour)
# print('chain log likelihoods', mcmc_chain.loglikelihood[:n_delete][chain_idx])
print('mcmc estimation of compression:', jnp.mean(first_iteration_loglikelihoods <= first_contour),
      'estimated from beta:', np.exp(log_t1))

print('n_live', n_live, 'n_delete', n_delete)
print('original evidence', np.mean(samples.logZ()))  # , '+-', np.sqrt(samples.D_KL()/n_live))
corrected_logX = logX

# Overwrite the logX method on this instance
samples.logX = types.MethodType(corrected_logX_fn, samples)
print('corrected evidence', samples.logZ())
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
print(f'real chain {chain_idx} start loc',
      samples[samples['pid'] == mcmc_chain_start_pid[chain_idx]][['c', 'm', 'sigma']].values)
# print('chain start', first_chain_position)
print(f'real chain {chain_idx} end loc',
      samples[samples['pid'] == mcmc_chain_end_pid[chain_idx]][['c', 'm', 'sigma']].values)
# print('chain last', last_chain_position)

real_start = samples[samples['pid'] == mcmc_chain_start_pid[chain_idx]][['c', 'm', 'sigma']].values[0]
real_end = samples[samples['pid'] == mcmc_chain_end_pid[chain_idx]][['c', 'm', 'sigma']].values[0]
full_c_chain = np.concatenate([np.array([real_start[0]]), mcmc_chain.position['c'][chain_idx]])
print('chain length', len(full_c_chain))
print('new first c', full_c_chain[0], 'new last c', full_c_chain[-1])
print('real first c', real_start[0], 'real last c', real_end[0])
