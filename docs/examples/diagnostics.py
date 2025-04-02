import numpy as np


def nested_rhat(samples: np.ndarray, K: int) -> float:
    """
    Compute the nested R̂ (potential scale reduction factor) diagnostic.

    This function assumes that the input `samples` is a NumPy array of shape (n, m),
    where n is the number of samples per short chain and m is the number of short chains.
    It computes:
      1. The usual R̂ over the m short chains.
      2. A weighted R̂ over K superchains formed by partitioning the m chains.

    The nested R̂ is defined as the maximum of the two R̂ values.

    Parameters:
        samples (np.ndarray): Array of shape (n, m) containing the MCMC samples.
        K (int): Number of superchains to create from the m short chains.

    Returns:
        float: The nested R̂ diagnostic.
    """
    n, m = samples.shape

    def compute_rhat(chains: np.ndarray) -> float:
        """
        Compute the standard R̂ for a set of chains.

        chains is an array of shape (L, m_local) where L is the length of each chain.
        """
        L, m_local = chains.shape
        chain_means = chains.mean(axis=0)
        chain_vars = chains.var(axis=0, ddof=1)
        overall_mean = chain_means.mean()
        # Between-chain variance (scaled by chain length)
        B = L * np.var(chain_means, ddof=1)
        # Within-chain variance (average of the chain variances)
        W = chain_vars.mean()
        # Estimate of the marginal posterior variance
        var_hat = ((L - 1) / L) * W + (B / L)
        return np.sqrt(var_hat / W)

    # Compute R̂ for the short chains.
    rhat_chains = compute_rhat(samples)

    # Partition the m chains into K groups (superchains).
    # We use np.array_split so that groups are as equal as possible.
    chain_indices = np.array_split(np.arange(m), K)
    superchains = []
    for idx in chain_indices:
        # Extract the group of chains and flatten into one long chain.
        group = samples[:, idx]  # shape (n, group_size)
        group_flat = group.flatten()
        superchains.append(group_flat)

    # Compute a weighted R̂ over the superchains.
    # For each superchain, compute its mean, variance, and length.
    means = np.array([sc.mean() for sc in superchains])
    vars_ = np.array([sc.var(ddof=1) for sc in superchains])
    lengths = np.array([len(sc) for sc in superchains])
    total_length = lengths.sum()
    # Compute the overall mean weighted by chain length.
    overall_mean_super = np.sum(means * lengths) / total_length
    # Between-superchain variance (weighted by lengths)
    B_super = np.sum(lengths * (means - overall_mean_super) ** 2) / (K - 1)
    # Within-superchain variance (pooled)
    W_super = np.sum((lengths - 1) * vars_) / (total_length - K)
    # Estimate of the marginal posterior variance for the superchains
    var_hat_super = ((total_length - 1) / total_length) * W_super + (B_super / total_length)
    rhat_super = np.sqrt(var_hat_super / W_super)

    # The nested R̂ is the maximum of the two R̂ values.
    return max(rhat_chains, rhat_super)


# Example usage:
if __name__ == "__main__":
    # Simulate some sample data: 16 samples per chain, 500 chains.
    np.random.seed(42)
    simulated_samples = np.random.normal(0, 1, size=(1000   , 500))
    K = 10  # For instance, partition the chains into 10 superchains.
    result = nested_rhat(simulated_samples, K)
    print("Nested R̂:", result)
