from scipy.stats import norm, expon
import scipy.integrate as integrate
import numpy as np
from scipy.special import comb
from scipy.special import gammainc


def block_er_ss2(l, M, L, gs):
    return 1 - (1 - (1 / M)**l * comb(gs * M, 2)) ** L

def mu(g, gamma, nu, theta, gstar, M):
    x0 = 0
    return mu_x0(g, gamma, nu, theta, gstar, M, x0)

def mu_x0(g, gamma, nu, theta, gstar, M, fr=0):
    to = gamma * (nu**-0.5) * (M**0.5) * (gstar - g)
    func = lambda x: norm.cdf(x) * np.exp(0.5 * np.power(x, 2))
    scalar = ((2 * np.pi)**0.5) / theta
    # print(g,gamma,nu,theta,gstar,M)
    integrval = integrate.quad(func, fr, to)
    # print(integrval)
    return scalar * integrval[0]

def pb_initpos(g, gamma, nu, theta, gstar, taustar, endstar, M, L):
    x0_mean = 0
    x0_var = nu / M
    x0_stdev = np.sqrt(x0_var)
    x0_to = gamma * (nu**-0.5) * (M**0.5) * (gstar - g)
    x0_fr = -1 * x0_to
    perr_x0 = lambda x0: pb_x0(g, gamma, nu, theta, gstar, taustar, endstar, M, L, x0)
    perr_x0_averaged = lambda x0: perr_x0(x0) * norm.pdf(x0, x0_mean, x0_stdev)
    perr = integrate.quad(perr_x0_averaged, x0_fr, x0_to)[0]
    return perr

def pb_correction(beta, g, gamma, nu, theta, gstar, taustar, endstar, M, L):
    if g > gstar:
        return 0
    alpha = np.sqrt(nu) / gamma
    #n = M * L
    n = M
    shift = beta * n**(-1/6) / np.sqrt(2 * np.pi * alpha ** 2) * np.exp(-n * (gstar - g)**2 / 2 / alpha**2)
    # power = 1/3
    # amu = mu(g, gamma, nu, theta, gstar, M)
    # shift = beta * n**power * amu**-1 * np.exp(-amu**-1 * (gstar - g))
    # perr = pb(g, gamma, nu, theta, gstar, taustar, endstar, M, L)
    # shift = beta * perr
    if shift > 1: shift = 1
    #print(shift)
    return shift

def pb(g, gamma, nu, theta, gstar, taustar, endstar, M, L):
    # print(g,gamma,nu,theta, gstar, taustar, M, L)
    x0 = 0
    return pb_x0(g, gamma, nu, theta, gstar, taustar, endstar, M, L, x0)

def pb_x0(g, gamma, nu, theta, gstar, taustar, endstar, M, L, x0=0):
    return 1 - np.exp(-1 * (endstar - taustar) / mu_x0(g, gamma, nu, theta, gstar, M, x0))

#def pb_corrected(beta, g, gamma, nu, theta, gstar, btime, taustar, endstar, M, L):
def pb_corrected(beta, g, gamma, nu, theta, gstar, taustar, endstar, M, L):
    #pb_uncorrected = pb(g, gamma, nu, theta, gstar, taustar, endstar, M, L)
    #pb_correction_term = pb_correction(beta, g, gamma, nu, theta, btime, taustar, endstar, M, L)
    #return pb_uncorrected
    #return 1 - (1 - pb_uncorrected)*(1 - pb_correction_term)
    #return pb_uncorrected + pb_correction_term
    correction = beta * M**(-2/3)
    if g + correction > gstar:
        correction = 0
    new_g = correction + g
    return pb(new_g, gamma, nu, theta, gstar, taustar, endstar, M, L)
    #return pb(g, gamma, nu, theta, gstar - beta * M**(-2/3), taustar, endstar, M, L)

def pb_term(g, gamma, nu, theta, gstar, taustar, endstar, M, L):
    amu = mu(g, gamma, nu, theta, gstar, M)
    # pdf = lambda x: amu**-2 * x * np.exp(-amu**-1 * x)
    # exp_n = g * L
    # func = lambda x: pdf(x)
    # ber = integrate.quad(func, 0, endstar - taustar)[0]
    # return ber
    end = endstar - taustar
    # return gammainc(2, end / amu)
    param = end / amu
    return 1 - (param + 1) * np.exp(-param)

#def pb_term_corrected(beta, g, gamma, nu, theta, gstar, btime, taustar, endstar, M, L):
def pb_term_corrected(beta, g, gamma, nu, theta, gstar, taustar, endstar, M, L):
    #pb_unc = pb_term(g, gamma, nu, theta, gstar, taustar, endstar, M, L)
    #shift = pb_correction(beta, g, gamma, nu, theta, btime, taustar, endstar, M, L)
    #ber = pb_unc + (1 - (1 - shift)**2)
    #ber = 1 - (1 - pb_unc)*((1-shift)**2)
    #ber = pb_unc + shift
    #return ber
    # correction = beta * M**(-2/3)
    # if g + correction > gstar:
    #     correction = 0
    # new_g = correction + g
    # new_taustar = taustar
    new_g = g
    correction = beta * M**(-1/3)
    new_taustar = taustar - correction
    return pb_term(new_g, gamma, nu, theta, gstar, new_taustar, endstar, M, L)

def plr(g, gamma, nu, theta, gstar, taustar, endstar, M, L):
    amu = mu(g, gamma, nu, theta, gstar, M)
    pdf = lambda x: expon.pdf(x, loc=taustar, scale=amu)
    exp_n = g * L
    func = lambda x: (1 - x / exp_n) * pdf(x)
    plr_limited = integrate.quad(func, taustar, endstar)[0]
    return plr_limited


def plr_term(g, gamma, nu, theta, gstar, taustar, endstar, M, L):
    amu = mu(g, gamma, nu, theta, gstar, M)
    pdf = lambda x: amu**-2 * x * np.exp(-amu**-1 * x)
    exp_n = g * L
    func = lambda x: (1 - x / exp_n) * pdf(x - taustar)
    plr_limited = integrate.quad(func, taustar, endstar)[0]
    return plr_limited

# the only difference between CSA and LDPC in terms of PLR is that
# in LDPC all the users that were not erased are immediately recovered
def plr_ldpc(g, gamma, nu, theta, gstar, taustar, endstar, M, L):
    amu = mu(g, gamma, nu, theta, gstar, M)
    # pdf = lambda x: expon.pdf(x, loc=taustar, scale=amu)
    # total = L
    # rec = (1 - g) * L
    # func = lambda x: (1 - (x + rec) / total) * pdf(x)
    # plr_limited = integrate.quad(func, taustar, endstar)[0]
    # TODO: test if the result is the same!
    dist = endstar - taustar
    plr_limited = amu * np.exp(-1*dist/amu) - amu + dist
    plr_limited /= L
    return plr_limited


def plr_ldpc_corrected(beta, g, gamma, nu, theta, gstar, taustar, endstar, M, L):
    correction = beta * M**(-2/3)
    if g + correction > gstar:
        correction = 0
    new_g = correction + g
    return plr_ldpc(new_g, gamma, nu, theta, gstar, taustar, endstar, M, L)

def plr_term_ldpc(g, gamma, nu, theta, gstar, taustar, endstar, M, L):
    amu = mu(g, gamma, nu, theta, gstar, M)
    # pdf = lambda x: amu**-2 * x * np.exp(-amu**-1 * x)
    # exp_n = g * L
    # total = L
    # rec = (1 - g) * L
    # func = lambda x: (1 - (x + rec) / total) * pdf(x - taustar)
    # plr_limited = integrate.quad(func, taustar, endstar)[0]
    #sol = lambda x: np.exp((taustar - x)/amu) * (amu * (2*amu - taustar) - x * (-2*amu + L*g + taustar) + L*g*(taustar - amu) + x**2)
    # sol = lambda x: np.exp((taustar - x)/amu) * ((x + amu)**2 + (amu - taustar)*(amu - L*g) - x*(L*g + taustar))
    # plr_limited = sol(endstar) - sol(taustar)
    # plr_limited /= L * amu
    #return plr_limited
    # PLR expression directly as in the paper.
    return np.exp((taustar - endstar)/amu) * (endstar**2 + taustar * g * L - (g*L + taustar - 2*amu)*(endstar + amu)) / (amu * L) + (g*L - taustar - 2*amu)/L

def plr_term_ldpc_corrected(beta, g, gamma, nu, theta, gstar, taustar, endstar, M, L):
    # correction = beta * M**(-2/3)
    # if g + correction > gstar:
    #     correction = 0
    # new_g = correction + g
    # new_taustar = taustar
    new_g = g
    correction = beta * M**(-1/3)
    new_taustar = taustar - correction
    return plr_term_ldpc(new_g, gamma, nu, theta, gstar, new_taustar, endstar, M, L)

def bler_term_ldpc_nofloor(g, gamma, nu, theta, gstar, taustar, endstar, v, M, L):
    m = mu(g, gamma, nu, theta, gstar, M)
    A = (endstar - taustar) / m

    pb = pb_term(g, gamma, nu, theta, gstar, taustar, endstar, M, L)
    ndec = m * v * (-np.exp(-A) * (A**2 + 2*A + 2) + 2)
    #ndec += L * (1 - pb)  # If the decoding is successful, we have recovered L blocks

    bler = pb - ndec / L
    return bler

def bler_ldpc_nofloor(g, gamma, nu, theta, gstar, taustar, endstar, v, M, L):
    m = mu(g, gamma, nu, theta, gstar, M)
    A = (endstar - taustar) / m

    pf = pb(g, gamma, nu, theta, gstar, taustar, endstar, M, L)
    ndec = m * v * (-np.exp(-A) * (A + 1) + 1)
    #ndec += L * (1 - pf)  # If the decoding is successful, we have recovered L blocks

    bler = pf - ndec / L
    return bler

def bler_term_ldpc_floor(g, gamma, nu, theta, gstar, taustar, endstar, v, M, L):
    m = mu(g, gamma, nu, theta, gstar, M)
    pb = pb_term(g, gamma, nu, theta, gstar, taustar, endstar, M, L)

    iss = np.arange(0, int((endstar - taustar) * v))
    ndec = 0
    for i in iss:
        prdec = np.exp(-i / v / m) * (i / v / m + 1) - np.exp(-(i+1) / v / m) * ((i + 1) / v / m + 1)
        ndec += i * prdec
    #ndec += L * (1 - pb)  # If the decoding is successful, we have recovered L blocks

    bler = pb - ndec / L
    return bler

def bler_ldpc_floor(g, gamma, nu, theta, gstar, taustar, endstar, v, M, L):
    m = mu(g, gamma, nu, theta, gstar, M)
    pf = pb(g, gamma, nu, theta, gstar, taustar, endstar, M, L)

    iss = np.arange(0, int((endstar - taustar) * v) - 1)
    width = endstar - taustar
    iwidth = int((endstar - taustar) * v)
    ndec = 0
    for i in iss:
        prdec = np.exp(-i / v / m) - np.exp(-(i+1) / v / m)
        ndec += i * prdec
    add = np.exp(-int((endstar - taustar) * v) / v / m) - np.exp(-width / m)
    ndec += iwidth * add
    #ndec += L * (1 - pf)  # If the decoding is successful, we have recovered L blocks

    bler = pf - ndec / L
    return bler

def bler_term_ldpc(g, gamma, nu, theta, gstar, taustar, endstar, v, M, L):
    #return bler_term_ldpc_floor(g, gamma, nu, theta, gstar, taustar, endstar, v, M, L)
    return bler_term_ldpc_nofloor(g, gamma, nu, theta, gstar, taustar, endstar, v, M, L)

def bler_ldpc(g, gamma, nu, theta, gstar, taustar, endstar, v, M, L):
    return bler_ldpc_floor(g, gamma, nu, theta, gstar, taustar, endstar, v, M, L)
    #return bler_ldpc_nofloor(g, gamma, nu, theta, gstar, taustar, endstar, v, M, L)
