import jax
import jax.numpy as jnp

from blackjax.ns.base import NSInfo


def log1mexp(x):
    """
    Numerically stable calculation of the quantity
    :math:`\\log(1 - \\exp(x))`, following the algorithm
    of `Mächler 2012`_.
    .. _Mächler 2012: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    Returns ``-jnp.inf`` when ``x == 0`` and ``jnp.nan``
    when ``x > 0``.
    :param x: A number or array of numbers.
    :return: The value of :math:`\\log(1 - \\exp(x))`.
    """
    return jnp.where(
        x > -0.6931472,  # approx log(2)
        jnp.log(-jnp.expm1(x)),
        jnp.log1p(-jnp.exp(x)),
    )


def compute_nlive(info: NSInfo):
    """
    Compute the effective number of live points at each death contour.

    Parameters
    ----------
    info : NSInfo
        Contains log-likelihood arrays for birth and death contours.

    Returns
    -------
    nlive : jnp.array
        Number of live points at each contour.
    """
    birth = info.logL_birth
    death = info.logL

    # Combine birth and death arrays
    combined = jnp.concatenate(
        [
            jnp.column_stack((birth, jnp.ones_like(birth))),
            jnp.column_stack((death, -jnp.ones_like(death))),
        ]
    )
    sorted_indices = jnp.lexsort((combined[:, 1], combined[:, 0]))
    sorted_combined = combined[sorted_indices]
    # cumsum = jnp.cumsum(sorted_combined[:, 1])
    cumsum = jnp.maximum(jnp.cumsum(sorted_combined[:, 1]), 0)

    death_mask = sorted_combined[:, 1] == -1
    nlive = cumsum[death_mask] + 1

    return nlive


def compute_log_compression(key: jax.random.PRNGKey, dead: NSInfo, samples=100):
    key, subkey = jax.random.split(key)
    min_val = jnp.finfo(dead.logL.dtype).tiny
    r = jnp.log(
        jax.random.uniform(subkey, shape=(dead.logL.shape[0], samples)).clip(
            min_val, 1 - min_val
        )
    )
    nlive = compute_nlive(dead)
    t = r / nlive[:, jnp.newaxis]
    return t


def logX(key: jax.random.PRNGKey, dead: NSInfo, samples=100, t=None):
    """Compute the log of the prior volume within iso-likelihood contours.

    This function calculates the log volume of the prior contained within
    each iso-likelihood contour for a nested sampling result.

    Parameters
    ----------
    key : jax.random.PRNGKey
        a jax rng key.
    dead : NSInfo
        An object containing the log-likelihoods and other relevant
        information of the dead points in nested sampling.
    samples : int, optional
        The number of samples to draw. Default is 100.
    t: double, optional
        log compression factors. Will compute them again if None are passed

    Returns
    -------
    logX : jnp.ndarray
        Cumulative log volume for each sample.
    logdX : jnp.ndarray
        Logarithm of the difference in volume for each contour.
    """
    if t is None:
        t = compute_log_compression(key, dead, samples)
    logX = jnp.cumsum(t, axis=0)
    return logX, logdX(logX)


def logdX(logX):
    """TODO"""
    logXp = jnp.concatenate([jnp.zeros((1, logX.shape[1])), logX[:-1]], axis=0)
    logXm = jnp.concatenate([logX[1:], jnp.full((1, logX.shape[1]), -jnp.inf)], axis=0)
    log_diff = logXm - logXp
    logdX = log1mexp(log_diff) + logXp - jnp.log(2)
    return logdX

def log_weights(key: jax.random.PRNGKey, dead: NSInfo, samples=100, beta=1.0):
    """
    Calculate the log importance weights for Nested Sampling results.

    Parameters
    ----------
    key : jax.random.PRNGKey
        a jax rng key.
    dead : NSInfo
        An object containing the log-likelihoods and other relevant
        information of the dead points in nested sampling.
    samples : int, optional
        The number of samples to draw for estimating log weights, by default 100.
    beta : float, optional
        The inverse temperature of the log-likelihood to calculate at,
        by default 1.0.

    Returns
    -------
    jnp.ndarray
        An array containing the log weights of the dead points.
    """
    # sort by logL
    j = jnp.argsort(dead.logL)
    original_indices = jnp.arange(len(dead.logL))
    dead = jax.tree.map(lambda x: x[j], dead)
    _, ldX = logX(key, dead, samples)
    ln_w = ldX + beta * dead.logL[..., jnp.newaxis]
    return ln_w[original_indices]


def finalise(state, dead):
    dead_map = jax.tree.map(
        lambda *args: jnp.concatenate(args),
        *(
                dead
                + [
                    NSInfo(
                        state.sampler_state.particles,
                        state.sampler_state.logL,
                        state.sampler_state.logL_birth,
                        state.sampler_state.pid,
                        dead[-1].update_info,
                        dead[-1].mcmc_chain,
                    )
                ]
        ),
    )
    return dead_map


def ess(rng_key, dead_map):
    logw = log_weights(rng_key, dead_map).mean(axis=-1)
    logw -= logw.max()
    l_sum_w = jax.scipy.special.logsumexp(logw)
    l_sum_w_sq = jax.scipy.special.logsumexp(2 * logw)
    ess = jnp.exp(2 * l_sum_w - l_sum_w_sq)
    return ess

def estimate_autocorrelation(TODO):
    # step 1: estimate traditionally
    # step 2: setup model AR(2)
    # step 3: estimate from all chains
    tau = None
    return tau


def sample(rng_key, dead_map, n=1000):
    logw = log_weights(rng_key, dead_map).mean(axis=-1)
    indices = jax.random.choice(
        rng_key,
        dead_map.logL.shape[0],
        p=jnp.exp(logw.squeeze() - jnp.max(logw)),
        shape=(n,),
        replace=True,
    )
    return jax.tree_util.tree_map(lambda leaf: leaf[indices], dead_map.particles)


def logZ(key: jax.random.PRNGKey, dead: NSInfo, samples=100, beta=1.0):
    """
    Compute the log evidence (log Z) from nested sampling dead points.

    This function estimates the log evidence by Monte Carlo integration
    over the iso-likelihood contours. It uses the log prior volume differences
    computed by `logX` and the corresponding log-likelihoods.

    Parameters
    ----------
    key : jax.random.PRNGKey
        A JAX random key.
    dead : NSInfo
        An object containing the nested sampling dead points, including
        log-likelihood values.
    samples : int, optional
        Number of Monte Carlo samples to draw per dead point, by default 100.
    beta : float, optional
        The inverse temperature, scaling the log-likelihood (default 1.0).

    Returns
    -------
    logZ : jnp.ndarray
        The estimated log evidence.
    """
    logw = log_weights(key, dead, samples=samples, beta=beta)
    return jax.scipy.special.logsumexp(logw, axis=0)

def logZ_mcmc(dead: NSInfo, n_delete, beta=1.0):
    j = jnp.argsort(dead.logL)
    original_indices = jnp.arange(len(dead.logL))
    sorted_dead = jax.tree.map(lambda x: x[j], dead)
    nlive = compute_nlive(sorted_dead)
    t = -1 / nlive[:, jnp.newaxis]
    skilling_logX = jnp.cumsum(t, axis=0)

    contours = jnp.sort(jnp.unique(dead.logL_birth[jnp.isfinite(dead.logL_birth)]))
    contour = contours[1]
    # print('chain likelihoods', dead.mcmc_chain.loglikelihood.shape, dead.mcmc_chain.loglikelihood)
    chain_likelihoods = jnp.ravel(dead.mcmc_chain.loglikelihood[:n_delete, 1:])
    # print('raveled chain likelihoods', chain_likelihoods.shape, chain_likelihoods)
    X_mcmc = jnp.mean(chain_likelihoods <= contour)
    logX_mcmc = jnp.log(X_mcmc)
    mcmc_logX = jnp.copy(skilling_logX)

    levels = jnp.arange(skilling_logX.shape[0])
    f = jnp.minimum(1.0, levels/n_delete)
    delta = (logX_mcmc - skilling_logX[n_delete])[0]
    mcmc_logX = skilling_logX + (f * delta)[:,jnp.newaxis]
    mcmc_logdX = logdX(mcmc_logX)
    logw = mcmc_logdX + beta * sorted_dead.logL[..., jnp.newaxis]
    logZ =  jax.scipy.special.logsumexp(logw[original_indices], axis=0)[0]
    err = logZ / jnp.sqrt(n_delete)
    return logZ, err
