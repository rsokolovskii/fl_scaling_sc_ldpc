# Simulating Ornstein-Uhlenbeck process
#
# Adapted from: https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
#
# Roman Sokolovskii

import numpy as np

def simulate_ou_trajectories(sigma, mu, tau, dt, T, ntrials, is_randomising):
    n = int(T / dt)  # Number of time steps.
    x = np.zeros((ntrials, n)) + mu
    if is_randomising:
        x[:,0] = sigma * np.random.randn(ntrials) + mu
    sigma_bis = sigma * np.sqrt(2. / tau)
    sqrtdt = np.sqrt(dt)
    for i in range(n - 1):
        x[:, i + 1] = x[:, i] + dt * (-(x[:, i] - mu) / tau) + \
            sigma_bis * sqrtdt * np.random.randn(ntrials)
    return x

def first_hit_times(trajectories, value):
    size = trajectories.shape[0]
    hits = np.zeros(size)
    last = trajectories.shape[1] - 1
    for i, tr in enumerate(trajectories):
        idxs, = np.where(tr >= value)
        hits[i] = np.min(idxs) if len(idxs) > 0 else last
    return hits

# Assumes the threshold is zero and the mean is larger than that
def first_hit_times_zero(trajectories):
    size = trajectories.shape[0]
    hits = np.zeros(size)
    last = trajectories.shape[1] - 1
    for i, tr in enumerate(trajectories):
        idxs, = np.where(tr <= 0)
        hits[i] = np.min(idxs) if len(idxs) > 0 else last
    return hits
