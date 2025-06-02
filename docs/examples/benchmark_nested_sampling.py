import jax
import jax.numpy as jnp
import tqdm
from jax.scipy.linalg import inv, solve

import blackjax
from blackjax.ns.utils import logZ, finalise, logZ_mcmc as logZ_from_mcmc

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
import matplotlib.pyplot as plt



def run(i,
        d,
        prior,
        loglikelihood):
    rng_key = jax.random.PRNGKey(i)
    n_live = 10
    n_delete = int(n_live/2)
    num_mcmc_steps = d * 5

    algo = blackjax.ns.adaptive.nss(
        logprior_fn=prior,
        loglikelihood_fn=loglikelihood,
        n_delete=n_delete,
        num_mcmc_steps=num_mcmc_steps,
    )

    rng_key, init_key, sample_key = jax.random.split(rng_key, 3)

    initial_particles = jax.random.multivariate_normal(init_key, prior_mean, prior_cov, (n_live,))
    state = algo.init(initial_particles, loglikelihood)


    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point

    dead = []
    with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
        while not state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)  # Update progress bar

    # It is now not too bad to remap the list of NSInfos into a single instance
    # note in theory we should include the live points, but assuming we have done things correctly and hit the termination criteria,
    # they will contain negligible weight
    #dead = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)
    dead = finalise(state, dead)

    # From here we can use the utils to compute the log weights and the evidence of the accumulated dead points
    # sampling log weights lets us get a sensible error on the evidence estimate
    logZ_skilling = logZ(rng_key, dead, samples=1000)
    logZ_mcmc, logZ_mcmc_err = logZ_from_mcmc(dead, n_delete=n_delete)

    # print(f"Runtime evidence: {state.sampler_state.logZ:.2f}")  # type: ignore[attr-defined]
    print(i)
    print(f"Estimated evidence: {logZ_skilling.mean():.4f} +- {logZ_skilling.std():.4f}")
    print(f"Estimated evidence with MCMC volume correction: {logZ_mcmc:.4f} +- {logZ_mcmc_err:.4f}")
    return logZ_skilling.mean(), logZ_mcmc

if __name__ == '__main__':
    rng_key = jax.random.PRNGKey(0)
    d = 2
    C = jax.random.normal(rng_key, (d, d)) * 0.1
    like_cov = C @ C.T
    like_mean = jax.random.normal(rng_key, (d,))
    prior_mean = jnp.zeros(d)
    prior_cov = jnp.eye(d) * 1
    prior = lambda x: jax.scipy.stats.multivariate_normal.logpdf(x, prior_mean, prior_cov)


    def loglikelihood(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=like_mean, cov=like_cov)


    def compute_logZ(mu_L, Sigma_L, logLmax=0, mu_pi=None, Sigma_pi=None):
        Sigma_P = inv(inv(Sigma_pi) + inv(Sigma_L))
        mu_P = jnp.dot(Sigma_P, (solve(Sigma_pi, mu_pi) + solve(Sigma_L, mu_L)))
        logdet_Sigma_P = jnp.linalg.slogdet(Sigma_P)[1]
        logdet_Sigma_pi = jnp.linalg.slogdet(Sigma_pi)[1]

        return (
                logLmax
                + logdet_Sigma_P / 2
                - logdet_Sigma_pi / 2
                - jnp.dot((mu_P - mu_pi), solve(Sigma_pi, mu_P - mu_pi)) / 2
                - jnp.dot((mu_P - mu_L), solve(Sigma_L, mu_P - mu_L)) / 2
        )


    log_analytic_evidence = compute_logZ(
        like_mean,
        like_cov,
        mu_pi=prior_mean,
        Sigma_pi=prior_cov,
        logLmax=loglikelihood(like_mean),
    )
    print(f"Analytic evidence: {log_analytic_evidence:.4f}")

    logZ_skilling = []
    logZ_mcmc = []
    for i in range(1000):
        val = run(i, d, prior, loglikelihood)
        logZ_skilling.append(val[0])
        logZ_mcmc.append(val[1])

    plt.hist(logZ_skilling, density=True, label='skilling', alpha=0.5)
    plt.hist(logZ_mcmc, density=True, label='MCMC', alpha=0.5)
    plt.axvline(log_analytic_evidence, color='r', label="analytic")
    plt.legend()
    plt.show()
    plt.figure()
    plt.scatter(logZ_skilling, logZ_mcmc)
    plt.xlabel('logZ_skilling')
    plt.ylabel('logZ_mcmc')
    plt.axvline(log_analytic_evidence, color='r', label="analytic")
    plt.axhline(log_analytic_evidence, color='r')
    plt.show()
