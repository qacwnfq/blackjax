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
# or the logX
logX = samples.logX()
# first compression factor is estimated right now from beta distribution:
log_t1 = logX.iloc[1] - logX.iloc[0]
print('log_t1', log_t1)

# estimating t1 using mcmc samples with the loglikelihoods of the mcmc samples -- this is likely the one you need
print('dead mcmc chain loglikelihood shape', dead.mcmc_chain.loglikelihood.shape)
# take the dead points from the first iteration:
print('dead after first it')
# Q: is there a way to do this without losing named tuple access?
dead_after_first_it = dead[:n_delete]
logLs = dead_after_first_it[1]
mcmc_chain = dead_after_first_it[5]
mcmc_chain_start_pid = dead_after_first_it[6]
mcmc_chain_end_pid = dead_after_first_it[7]

print('mcmc_chain')
print(mcmc_chain)

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

print(first_chain_position)
print(last_chain_position)
print('start and end pids for chain ', chain_idx)
print(mcmc_chain_start_pid[chain_idx])
print(mcmc_chain_end_pid[chain_idx])
print(f'real chain {chain_idx} start loc', samples[samples['pid']==mcmc_chain_start_pid[chain_idx]][['c', 'm', 'sigma']].values)
print('chain start', first_chain_position)
print(f'real chain {chain_idx} end loc', samples[samples['pid']==mcmc_chain_end_pid[chain_idx]][['c', 'm', 'sigma']].values)
print('chain last', last_chain_position)
# A: the real chain start is missing from the mcmc chain (which is what we expect from this implementation of ns)


# log likelihoods
# print(logL_birth)
# print(logL)


exit(0)

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
