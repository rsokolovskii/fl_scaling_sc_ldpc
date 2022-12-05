import numpy as np
import matplotlib.pyplot as plt

def est_scaling_params_with_theta(r1s, r1s_theory, g, gstar, M, L):
    lvl, idxs = find_level(r1s_theory)
    gamma = calc_gamma(lvl, M, gstar, g)
    nu = calc_nu(r1s, r1s_theory, idxs, M)
    taustar_coeff = calc_taustar_coeff(idxs, M, L)
    taustar = calc_taustar(taustar_coeff, L)
    theta = calc_theta(r1s, idxs, M)
    end_coeff = calc_end_coeff(idxs, M, L)

    print(f"taustar = lambda g: {taustar_coeff} * L")
    print(f"end_coeff = lambda g: g * L")
    print(f"gamma   = {gamma}")
    print(f"nu      = {nu}")
    print(f"theta   = {theta}")
    return taustar, end_coeff, gamma, nu, theta

def est_scaling_params(r1s, r1s_theory, g, gstar, M, L):
    lvl, idxs = find_level(r1s_theory)
    gamma = calc_gamma(lvl, M, gstar, g)
    # nu = calc_nu_flattened(r1s, lvl, idxs, M)
    nu = calc_nu(r1s, r1s_theory, idxs, M)
    taustar_coeff = calc_taustar_coeff(idxs, M, L)
    taustar = calc_taustar(taustar_coeff, L)
    #theta = calc_theta(r1s, idxs, M)
    end_coeff = calc_end_coeff(idxs, M, L)

    print(f"taustar = lambda g: {taustar_coeff} * L")
    print(f"end_coeff = lambda g: g * L")
    print(f"gamma   = {gamma}")
    print(f"nu      = {nu}")
    #print(f"theta   = {theta}")
    return taustar, end_coeff, gamma, nu, None


def find_level(r1s_theory):
    nz = r1s_theory[r1s_theory > 1]
    threshold = 1e-3
    idxs = np.abs(nz[:-1] - nz[1:]) < threshold
    lvls = nz[:-1][idxs]
    #plt.semilogy(r1s_theory)
    #plt.show()
    return np.mean(lvls), idxs

def find_level_robust(r1s_theory, M, estar, e, tolerance=1e-2):
    de = estar - e
    r1s_theory_norm = r1s_theory / M / de
    npoints = 1000
    ivl = int(r1s_theory_norm.shape[0] / npoints)
    r1s_theory_norm = r1s_theory_norm[::ivl]

    nz = r1s_theory_norm[r1s_theory_norm > 0]
    idxs = np.abs(nz[:-1] - nz[1:]) < tolerance
    lvls = nz[:-1][idxs]
    ssstart = np.min(np.where(idxs))
    ssstop = ssstart + np.min(np.where(np.abs(nz[ssstart:-1] - nz[ssstart+1:]) > tolerance))
    #idxs = np.arange(ssstart, ssstop)
    return ssstart * ivl / M, ssstop * ivl / M

def find_level_robust_ppd(r1s_theory, N, estar, e, tolerance=1e-2):
    de = estar - e
    r1s_theory_norm = r1s_theory / N / de

    nz = r1s_theory_norm[r1s_theory_norm > 0]
    idxs = np.abs(nz[:-1] - nz[1:]) < tolerance
    lvls = nz[:-1][idxs]
    ssstart = np.min(np.where(idxs))
    ssstop = ssstart + np.min(np.where(np.abs(nz[ssstart:-1] - nz[ssstart+1:]) > tolerance))
    #idxs = np.arange(ssstart, ssstop)
    return ssstart, ssstop

def calc_gamma(lvl, M, gstar, g):
    return lvl / M / (gstar - g)

# Old version; kept for compatibility with python notebooks.
# The new version is calc_nu_flattened below
def calc_nu(r1s, r1s_theory, idxs, M):
    varM = calc_var_biased(r1s[:,:np.max(np.where(r1s_theory > 1))+1], r1s_theory[r1s_theory > 1], M)
    varM_steady = varM[:-1][idxs]
    plt.plot(varM[:-1][idxs])
    plt.show()
    return np.mean(varM_steady)

def calc_nu_chunk(r1s_chunk, r1s_theory, M):
    r1s_chunk_cropped = r1s_chunk[:,:np.max(np.where(r1s_theory > 0))+1]
    r1s_theory_cropped = r1s_theory[r1s_theory > 0]
    r1s_chunk_cropped = r1s_chunk_cropped[:,0:r1s_theory.shape[0]]  # ... ?!
    return calc_var_chunk(r1s_chunk_cropped, r1s_theory_cropped, M)


def calc_nu_flattened(r1s, lvl, idxs, M):
    r1s = r1s[:,:len(idxs)]
    return calc_var_flattened(r1s[:,idxs], lvl, M)

# We ignore all the values below the mean
def calc_var(r1s_inp, lvl, frame_len):
    r1s = r1s_inp / frame_len
    r1_centered = r1s - lvl / frame_len
    r1_centered[r1_centered <= 0] = np.nan
    var = np.nanmean(r1_centered**2, axis=0)
    varM = var * frame_len
    return varM

# Not only we ignore all values below the mean, but we also
# aggregate all the samples across the steady state
def calc_var_flattened(r1s_inp, lvl, M):
    r1_theory = lvl / M
    r1s = r1s_inp / M
    r1_centered = r1s - r1_theory
    r1_centered[r1_centered <= 0] = np.nan
    var = np.nanmean(r1_centered**2)
    varM = var * M
    return varM

# We DO NOT ignore all the values below the mean
def calc_var_biased(r1s_inp, r1_theory_inp, frame_len):
    r1_theory = r1_theory_inp / frame_len
    r1s = r1s_inp[:,0:r1_theory.shape[0]] / frame_len
    r1_centered = r1s - r1_theory
    r1_centered[r1s == 0] = np.nan
    var = np.nanmean(r1_centered**2, axis=0)
    varM = var * frame_len
    return varM

def calc_var_chunk(r1s_inp, r1_theory_inp, M):
    r1_theory = r1_theory_inp / M
    r1s = r1s_inp / M
    r1_centered = r1s - r1_theory
    r1_centered[r1s == 0] = np.nan
    ssquares = np.nansum(r1_centered**2, axis=0)
    counts = np.sum(~np.isnan(r1_centered), axis=0)
    return ssquares, counts

def calc_taustar_coeff(idxs, M, L):
    start = np.min(np.where(idxs))
    #stop = np.max(np.where(idxs))
    taustar_coeff = start / M / L
    return taustar_coeff

def calc_end_coeff(idxs, M, L):
    stop = np.max(np.where(idxs))
    end_coeff = stop / M / L
    return end_coeff

def calc_taustar(taustar_coeff, L):
    return taustar_coeff * L

def calc_theta(r1s, idxs, M):
    start = np.min(np.where(idxs))
    stop = np.max(np.where(idxs))
    return calc_theta_explicit_ss_bounds(r1s, start, stop, M)

import pandas as pd
from scipy.optimize import curve_fit
def calc_theta_explicit_ss_bounds(r1s, start, stop, M):
    npoints = 1000
    ivl = int((stop - start) / npoints)
    ivl = 1
    
    r1s_float = np.array(r1s, dtype='float')
    r1s_float = r1s_float[:,start:stop:ivl] / M
    r1s_float[r1s_float == 0] = np.nan
    r1s_df = pd.DataFrame(r1s_float)

    c = r1s_df.corr()

    space = np.linspace(0.3, 0.7, 10)
    #space = [0.4, 0.6]
    thetas = []
    for l in space:
        loc = (start + (stop - start) * l)
        func = lambda x, theta: np.exp(-theta * np.abs(x - loc))
        xdata = np.arange(start,stop,ivl)
        ydata = c[int(c.shape[0] * l)]
    
        popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, 20))
        theta = tuple(popt)[0]
        thetas.append(theta)
        plt.plot(xdata, func(xdata, *popt))
        plt.plot(xdata, ydata)
    plt.show()

    #print(thetas)
    theta = np.mean(thetas)
    return theta