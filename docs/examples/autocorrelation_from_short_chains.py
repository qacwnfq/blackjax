import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(1234)

# Build the celerite model:
import celerite
from celerite import terms


def simulate_autocorrelations():
    kernel = terms.RealTerm(log_a=0.0, log_c=-6.0)
    kernel += terms.RealTerm(log_a=0.0, log_c=-2.0)

    # The true autocorrelation time can be calculated analytically:
    true_tau = sum(2 * np.exp(t.log_a - t.log_c) for t in kernel.terms)
    true_tau /= sum(np.exp(t.log_a) for t in kernel.terms)
    gp = celerite.GP(kernel)
    print('true tau is', true_tau)

    return true_tau, gp

def autocorr_ml(y, thin=1, c=5.0):
    # Compute the initial estimate of tau using the standard method
    init = autocorr_new(y, c=c)
    z = y[:, ::thin]
    N = z.shape[1]

    # Build the GP model
    tau = max(1.0, init / thin)
    kernel = terms.RealTerm(
        np.log(0.9 * np.var(z)),
        -np.log(tau),
        bounds=[(-5.0, 5.0), (-np.log(N), 0.0)],
    )
    kernel += terms.RealTerm(
        np.log(0.1 * np.var(z)),
        -np.log(0.5 * tau),
        bounds=[(-5.0, 5.0), (-np.log(N), 0.0)],
    )
    gp = celerite.GP(kernel, mean=np.mean(z))
    gp.compute(np.arange(z.shape[1]))

    # Define the objective
    def nll(p):
        # Update the GP model
        gp.set_parameter_vector(p)

        # Loop over the chains and compute likelihoods
        v, g = zip(*(gp.grad_log_likelihood(z0, quiet=True) for z0 in z))

        # Combine the datasets
        return -np.sum(v), -np.sum(g, axis=0)

    # Optimize the model
    p0 = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    soln = minimize(nll, p0, jac=True, bounds=bounds)
    gp.set_parameter_vector(soln.x)

    # Compute the maximum likelihood tau
    a, c = kernel.coefficients[:2]
    tau = thin * 2 * np.sum(a / c) / np.sum(a)
    return tau

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


if __name__ == "__main__":
    true_tau, gp = simulate_autocorrelations()
    print(true_tau)
    # simulate chains
    for n_chains in [5, 10, 25, 50, 100, 200, 500, 1000]:
        print('computing', n_chains)
        max_steps = int(true_tau*75)
        t = np.arange(max_steps)
        gp.compute(t)
        y = gp.sample(size=n_chains)

        # plt.plot(y[:n_chains, :n_steps*100].T)
        # plt.xlim(0, n_steps*100)
        # plt.xlabel("step number")
        # plt.ylabel("$f$")
        # plt.title("$\\tau_\mathrm{{true}} = {0:.0f}$".format(true_tau), fontsize=14)
        # plt.show()
        # Compute the estimators for a few different chain lengths
        N = np.exp(np.linspace(np.log(10), np.log(y.shape[1]), 25)).astype(int)
        ml = np.empty(len(N))
        ml[:] = np.nan
        new = np.empty(len(N))
        for i, n in enumerate(N):
            new[i] = autocorr_new(y[:, :n])

        for j, n in enumerate(N[1:len(N-1)]):
            i = j + 1
            thin = max(1, int(0.05 * new[i]))
            try:
                ml[i] = autocorr_ml(y[:, :n], thin=thin)
            except:
                pass

        plt.figure()
        plt.title(f'n_chains={n_chains}, true tau = {true_tau}')
        plt.loglog(N, new, "o-", label="emcee state-of-art")
        plt.loglog(N, ml, "o-", label="fitted autoregressive model")
        ylim = plt.gca().get_ylim()
        plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
        plt.axhline(true_tau, color='r', label=r'true $\tau$')
        plt.axvline(true_tau, color='r')
        plt.ylim(ylim)
        plt.xlabel("number of samples, $N$")
        plt.ylabel(r"$\tau$ estimates")
        plt.legend(fontsize=14)
        plt.savefig(f'/home/jadebeck/repos/blackjax/figs3/nchains={n_chains}.png')
        plt.savefig(f'/home/jadebeck/repos/blackjax/figs3/nchains={n_chains}.svg')
        plt.show()



