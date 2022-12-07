# Scaling laws for full BP decoding with limited iterations
#
# Described in
# Roman Sokolovskii, Alexandre Graell i Amat, Fredrik Brännström,
# "Finite-Length Scaling of SC-LDPC Codes With a Limited Number of Decoding Iterations"
# https://arxiv.org/abs/2203.08880
#
# Implemented by Roman Sokolovskii

from io import StringIO
from scipy.stats import norm
import scipy.integrate as integrate
from sklearn.neighbors import KernelDensity

from fl_scaling.ornstein_uhlenbeck import first_hit_times, first_hit_times_zero

def pb_term_bec_lim_iter_pdf_sim_ppd_internal(e, epdf_sim, width):
    def qfunc(z, x):
        q_a = epdf_sim(z)
        q_b = epdf_sim((x - z))
        return q_a * q_b
    qfunc_int = lambda x: integrate.quad(qfunc, 0, x, args=(x,))[0]
    def success_func(x):
        return qfunc_int(x)
    success_func_int = integrate.quad(success_func, width, np.inf)[0]
    fer = 1 - success_func_int
    print(e, fer)
    return fer


def make_pdf_sim(sim_vals):
    # fit density
    model = KernelDensity(bandwidth=0.4,kernel='gaussian')
    sample = np.reshape(sim_vals, (len(sim_vals), 1))
    model.fit(sample)

    def pdf_sim(x_th):
        x_th = np.asarray([x_th]) # NB: will fail if an array is passed as a parameter :(
        values = x_th.reshape((len(x_th), 1))
        probabilities = model.score_samples(values)
        probabilities = np.exp(probabilities)
        return probabilities
    return pdf_sim

def load_trajectories(e, N, num_files=200, pad_size=700):
    tr_array_list = []
    tr_vns_list = []

    for i in range(num_files):
        with open(f'cluster_sim/trajectories_bp/trajectories_{e:.4f}_truncated_SC_LDPC_5_10_L50_M{int(N/2)}_BP_Full_{pad_size}it_Random_BLER_{i}.dat', 'rt') as f_data:    
            for tr in f_data.read().split('\n\n'):
                if tr == "": continue
                tr_str = StringIO(tr)
                tr_iter, tr_arr, tr_vns = np.loadtxt(tr_str, unpack=True)
                tr_arr = np.pad(tr_arr, (0, pad_size - len(tr_arr)), mode="constant")
                tr_vns = np.pad(tr_vns, (0, pad_size - len(tr_vns)), mode="constant")
                tr_array_list.append(tr_arr)
                tr_vns_list.append(tr_vns)

    tr_deg1s = np.vstack([tr for tr in tr_array_list])
    tr_vns = np.vstack([tr for tr in tr_vns_list])
    return tr_vns, tr_deg1s

# pdf_sims = {}
# trajectories = {}
def pb_term_bec_lim_iter_pdf_sim_ppd(e, N, I):
    if (e, N, I) in pdf_sims:
        print("FOUND THE PDF SIM")
        pdf_sim = pdf_sims[(e, N, I)]
    else:
        if (e, N) in trajectories:
            print("FOUND THE TRAJECTORIES")
            trajs = trajectories[(e, N)]
        else:
            num_files = 100 if N == 1000 and (e == 0.465 or e == 0.46) else 200
            pad_size = 500 if (N == 1000 and e < 0.47) else 700
            trajs = load_trajectories(e, N, num_files, pad_size)[0]
            trajectories[(e, N)] = trajs

        init_period = int(round(float(taustar_interp_ldpc_term_ppd_vn(e))))
        trajs_ss = trajs[:, init_period : ]
        num_iter = int(round(I - taustar_interp_ldpc_term_ppd_vn(e) - collapse_interp_ldpc_term_ppd_vn(e)))
        cum_hist = np.sum(trajs_ss[:,0:num_iter], axis=1) / N
        pdf_sim = make_pdf_sim(cum_hist)

        pdf_sims[(e, N, I)] = pdf_sim

    #width = endstar_interp_ldpc_term(e) - taustar_interp_ldpc_term(e) 
    width = 2 * gammaf_ppd_vn(e) * (gstar - e) * (endstar_interp_ldpc_term_ppd_vn(e) - taustar_interp_ldpc_term_ppd_vn(e))
    return pb_term_bec_lim_iter_pdf_sim_ppd_internal(e, pdf_sim, width)


import scipy.stats as stats
from scipy.stats import norm
import scipy.integrate as integrate

from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.neighbors import KernelDensity

from fl_scaling.scaling_laws_full_BP_unlim_iter import *


import warnings
warnings.filterwarnings("ignore")

# https://mathoverflow.net/questions/84952/time-integral-of-an-ornstein-uhlenbeck-process
# https://mathoverflow.net/questions/254927/autocovariance-of-time-integrated-ornstein-uhlenbeck-process?rq=1
# Uhlenbeck, G. E.; Ornstein, L. S. (1930). "On the theory of Brownian Motion". Phys. Rev. 36 (5): 823–841.
def pb_term_bec_lim_iter_min_ppd_vn(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time, endstar_ppd_vn, gamma_ppd_vn, nu_ppd_vn, theta_ppd_vn, shiftf_ppd_vn):
    propagation_iter = I - init_period - collapse_time
    amu = mu(g, gamma, nu, theta, gstar, N)
    lam = 1 / amu
    width = endstar - taustar
    # if g > 0.48:
    #     corr = shiftf_ppd_vn(0.48) / 1000
    # else:
    #     corr = shiftf_ppd_vn(g) / 1000
    #width = 2 * (gamma_ppd_vn * (gstar - g) - corr) * (endstar_ppd_vn - init_period)
    #width = 2 * gamma_ppd_vn * (gstar - g) * (endstar_ppd_vn - init_period)
    loc = gamma_ppd_vn * (gstar - g) * propagation_iter
    #loc -= corr * propagation_iter

    # t_nr = propagation_iter
    # sigmasq = nu_ppd_vn / N * 2. * theta_ppd_vn * (gstar - g)
    # thet = theta_ppd_vn * (gstar - g)
    # var = sigmasq / (2 * thet**3) * (2 * t_nr * thet - 3 + 4*np.exp(-thet*t_nr) - np.exp(-2*thet*t_nr))
    # scale = np.sqrt(var)

    correction = np.sqrt(2 / theta_ppd_vn / (gstar - g))  # from an integrated OU process
    scale = np.sqrt(propagation_iter * nu_ppd_vn / N) * correction
    #scale = np.sqrt(propagation_iter * nu / N) * correction

    pdfmin = lambda x: stats.norm.pdf(x, loc=loc, scale=scale) * np.exp(-x * lam) +\
             stats.norm.sf(x, loc=loc, scale=scale) * lam * np.exp(-x * lam)
    def qfunc(z, x):
        q_a = pdfmin(z)
        q_b = pdfmin(x - z)
        return q_a * q_b
    qfunc_int = lambda x: integrate.quad(qfunc, 0, x, args=(x,))[0]
    def success_func(x):
        return qfunc_int(x)
    success_func_int = integrate.quad(success_func, width, np.inf)[0]
    fer = 1 - success_func_int
    print(g, fer)
    return fer


def estimate_ou_pd_mean(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time, amu=None):
    print(f"Estimating the OU jumping mean for epsilon={g}, N={N}, I={I}...")
    if amu is None:
        amu = mu(g, gamma, nu, theta, gstar, N)

    ### PARAMETERS IN LANGEVIN NOTATION
    mean = 0.0

    sigma = np.sqrt(nu / N)
    tau = 1 / theta
    dt = 1 / N
    #T = g * L - taustar
    T = L
    ntrials = 10000
    #ntrials = 100

    threshold = gamma * (gstar - g)
    trajectories_randomised = simulate_ou_trajectories(sigma, mean, tau, dt, T, ntrials, is_randomising=True)
    trajectories = trajectories_randomised
    print("Randomised OU simulation has been completed.")
    
    hs = first_hit_times(trajectories, threshold)

    # We'll throw away trajectories only if we overshoot the first-hit time
    traj_succ = trajectories # [np.where(hs == trajectories.shape[1] - 1)]
    traj_succ = traj_succ[:, :-1]
    traj_succ *= -1
    traj_succ += threshold

    propagation_iter = int(round(I - init_period - collapse_time))

    lens = []
    #lens = np.zeros(traj_succ.shape[0])
    for i, tr in enumerate(traj_succ):
        #assert np.all(tr > 0)
        x = 0
        abort = False
        for it in range(propagation_iter):
            if int(round(x * N)) >= hs[i]:
                abort = True
                break
            else:
                x += tr[int(round(x * N))]
        if not abort:
            lens.append(x)
        #lens[i] = x
    ou_mean = np.mean(lens)
    print("Empirical OU jumping mean has been calculated.")
    return ou_mean



# The location of the Gaussian is determined through the jumping OU PD model
def pb_term_bec_lim_iter_min_ppd_vn_jumping_ou_loc(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time, endstar_ppd_vn, gamma_ppd_vn, nu_ppd_vn, theta_ppd_vn):
    propagation_iter = I - init_period - collapse_time
    amu = mu(g, gamma, nu, theta, gstar, N)
    lam = 1 / amu
    width = endstar - taustar
    #width = 2 * (gamma_ppd_vn * (gstar - g) - corr) * (endstar_ppd_vn - init_period)
    #width = 2 * gamma_ppd_vn * (gstar - g) * (endstar_ppd_vn - init_period)
    #loc = gamma_ppd_vn * (gstar - g) * propagation_iter
    #loc -= corr * propagation_iter
    loc = estimate_ou_pd_mean(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time, amu)

    # t_nr = propagation_iter
    # sigmasq = nu_ppd_vn / N * 2. * theta_ppd_vn * (gstar - g)
    # thet = theta_ppd_vn * (gstar - g)
    # var = sigmasq / (2 * thet**3) * (2 * t_nr * thet - 3 + 4*np.exp(-thet*t_nr) - np.exp(-2*thet*t_nr))
    # scale = np.sqrt(var)

    correction = np.sqrt(2 / theta_ppd_vn / (gstar - g))  # from an integrated OU process
    scale = np.sqrt(propagation_iter * nu_ppd_vn / N) * correction
    #scale = np.sqrt(propagation_iter * nu / N) * correction

    pdfmin = lambda x: stats.norm.pdf(x, loc=loc, scale=scale) * np.exp(-x * lam) +\
             stats.norm.sf(x, loc=loc, scale=scale) * lam * np.exp(-x * lam)
    def qfunc(z, x):
        q_a = pdfmin(z)
        q_b = pdfmin(x - z)
        return q_a * q_b
    qfunc_int = lambda x: integrate.quad(qfunc, 0, x, args=(x,))[0]
    def success_func(x):
        return qfunc_int(x)
    success_func_int = integrate.quad(success_func, width, np.inf)[0]
    fer = 1 - success_func_int
    print(g, fer)
    return fer


# The location of the Gaussian is given from the jumping OU PD model (interpolated for different I)
def pb_term_bec_lim_iter_min_ppd_vn_jumping_ou_loc_interp(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time, endstar_ppd_vn, gamma_ppd_vn, nu_ppd_vn, theta_ppd_vn, loc):
    propagation_iter = I - init_period - collapse_time
    amu = mu(g, gamma, nu, theta, gstar, N)
    lam = 1 / amu
    width = endstar - taustar
    #width = 2 * (gamma_ppd_vn * (gstar - g) - corr) * (endstar_ppd_vn - init_period)
    #width = 2 * gamma_ppd_vn * (gstar - g) * (endstar_ppd_vn - init_period)

    # t_nr = propagation_iter
    # sigmasq = nu_ppd_vn / N * 2. * theta_ppd_vn * (gstar - g)
    # thet = theta_ppd_vn * (gstar - g)
    # var = sigmasq / (2 * thet**3) * (2 * t_nr * thet - 3 + 4*np.exp(-thet*t_nr) - np.exp(-2*thet*t_nr))
    # scale = np.sqrt(var)

    correction = np.sqrt(2 / theta_ppd_vn / (gstar - g))  # from an integrated OU process
    scale = np.sqrt(propagation_iter * nu_ppd_vn / N) * correction

    pdfmin = lambda x: stats.norm.pdf(x, loc=loc, scale=scale) * np.exp(-x * lam) +\
             stats.norm.sf(x, loc=loc, scale=scale) * lam * np.exp(-x * lam)
    def qfunc(z, x):
        q_a = pdfmin(z)
        q_b = pdfmin(x - z)
        return q_a * q_b
    qfunc_int = lambda x: integrate.quad(qfunc, 0, x, args=(x,))[0]
    def success_func(x):
        return qfunc_int(x)
    success_func_int = integrate.quad(success_func, width, np.inf)[0]
    fer = 1 - success_func_int
    print(g, fer)
    return fer

def estimate_ou_ecdf_ppd_vn(g, gamma, nu, theta, gstar, N, I, init_period, collapse_time):
    print(f"Estimating the ECDF and EPDF for epsilon={g}, N={N}, I={I}...")

    ### PARAMETERS IN LANGEVIN NOTATION
    mean = gamma * (gstar - g)

    sigma = np.sqrt(nu / N)
    tau = 1 / theta / (gstar - g)
    dt = 1
    T = int(round(I - init_period - collapse_time))
    ntrials = 10000

    trajectories_randomised = simulate_ou_trajectories(sigma, mean, tau, dt, T, ntrials, is_randomising=True)
    trajectories = trajectories_randomised
    print("Randomised OU simulation has been completed.")
    
    lens = np.sum(trajectories, axis=1)

    # sample probabilities for a range of outcomes
    if len(lens) == 0:
        ecdf = lambda x: 1
        epdf = lambda x: 0
        ecdf_conv = lambda x: 1
    else:
        ecdf = ECDF(lens)
        
        # fit density
        model = KernelDensity(bandwidth=0.1,kernel='gaussian')
        sample = lens.reshape((len(lens), 1))
        model.fit(sample)

        max_lens = np.max(lens)
        x_th = np.linspace(0, max_lens, 1000)
        values = x_th.reshape((len(x_th), 1))
        probabilities = model.score_samples(values)
        probabilities = np.exp(probabilities)

        prob_conv = np.convolve(probabilities, probabilities)
        prob_conv /= np.sum(prob_conv)
        x_conv = np.linspace(0, 2*max_lens, 1999)

        cdf_conv = np.cumsum(prob_conv)
        def ecdf_conv(threshold):
            if threshold > 2 * max_lens: return 1
            idx = np.argmax(x_conv >= threshold)
            cdf_point = cdf_conv[idx]
            return cdf_point
        def epdf(x_th):
            x_th = np.asarray([x_th]) # NB: will fail if an array is passed as a parameter :(
            values = x_th.reshape((len(x_th), 1))
            probabilities = model.score_samples(values)
            probabilities = np.exp(probabilities)
            return probabilities

    print("Empirical CDF and PDF have been calculated.")
    return ecdf, epdf, ecdf_conv


def pb_term_bec_lim_iter_min_ecdf_sim_ppd_vn(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time, gamma_ppd_vn, nu_ppd_vn, theta_ppd_vn):
    amu = mu(g, gamma, nu, theta, gstar, N)
    ecdf, epdf, _ecdf_conv = estimate_ou_ecdf_ppd_vn(g, gamma_ppd_vn, nu_ppd_vn, theta_ppd_vn, gstar, N, I, init_period, collapse_time)
    lam = 1 / amu
    width = endstar - taustar
    pdfmin = lambda x: epdf(x) * np.exp(-x * lam) +\
             (1 - ecdf(x)) * lam * np.exp(-x * lam)
    def qfunc(z, x):
        q_a = pdfmin(z)
        q_b = pdfmin(x - z)
        return q_a * q_b
    qfunc_int = lambda x: integrate.quad(qfunc, 0, x, args=(x,))[0]
    def success_func(x):
        return qfunc_int(x)
    success_func_int = integrate.quad(success_func, width, np.inf)[0]
    fer = 1 - success_func_int
    print(g, fer)
    return fer

def calc_epdf_ecdf(data, bandwidth=0.1):
    if len(data) == 0:
        ecdf = lambda x: 1
        epdf = lambda x: 0
        ecdf_conv = lambda x: 1
    else:
        ecdf = ECDF(data)
        # fit density
        model = KernelDensity(bandwidth=bandwidth,kernel='gaussian')
        sample = data.reshape((len(data), 1))
        model.fit(sample)

        max_lens = np.max(data)
        x_th = np.linspace(0, max_lens, 1000)
        values = x_th.reshape((len(x_th), 1))
        probabilities = model.score_samples(values)
        probabilities = np.exp(probabilities)

        prob_conv = np.convolve(probabilities, probabilities)
        prob_conv /= np.sum(prob_conv)
        x_conv = np.linspace(0, 2*max_lens, 1999)

        cdf_conv = np.cumsum(prob_conv)
        def ecdf_conv(threshold):
            if threshold > 2 * max_lens: return 1
            idx = np.argmax(x_conv >= threshold)
            cdf_point = cdf_conv[idx]
            return cdf_point
        def epdf(x_th):
            x_th = np.asarray([x_th]) # NB: will fail if an array is passed as a parameter :(
            values = x_th.reshape((len(x_th), 1))
            probabilities = model.score_samples(values)
            probabilities = np.exp(probabilities)
            return probabilities
    return epdf, ecdf, ecdf_conv

from fl_scaling.ornstein_uhlenbeck import simulate_ou_trajectories, first_hit_times_zero

def estimate_ou_ecdf_ppd_vn_absorbed(g, gamma, nu, theta, gstar, N, I, init_period, collapse_time, shift_ppd_vn=0.0):
    print(f"Estimating the ECDF and EPDF for epsilon={g}, N={N}, I={I}...")

    ### PARAMETERS IN LANGEVIN NOTATION
    mean = gamma * (gstar - g) - shift_ppd_vn

    sigma = np.sqrt(nu / N)
    tau = 1 / theta / (gstar - g)
    dt = 1
    T = int(round(I - init_period - collapse_time))
    ntrials = 100000

    trajectories_randomised = simulate_ou_trajectories(sigma, mean, tau, dt, T, ntrials, is_randomising=True)
    trajectories = trajectories_randomised
    print("Randomised OU simulation has been completed.")

    hit_times = first_hit_times_zero(trajectories)
    cum_hist_ou_absorbed = np.zeros(trajectories.shape[0])
    for i, tr in enumerate(trajectories):
        cum_hist_ou_absorbed[i] = np.sum(trajectories[i, 0 : int(hit_times[i])])

    lens = cum_hist_ou_absorbed
    epdf, ecdf, ecdf_conv = calc_epdf_ecdf(lens)

    print("Empirical CDF and PDF have been calculated.")
    return epdf, ecdf, ecdf_conv


def estimate_ou_ecdf_absorbed(g, gamma, nu, theta, gstar, N, L):
    print(f"Estimating the ECDF and EPDF for epsilon={g}, N={N}, L={L}...")

    ### PARAMETERS IN LANGEVIN NOTATION
    mean = gamma * (gstar - g)

    sigma = np.sqrt(nu / N)
    tau = 1 / theta
    dt = 1 / N
    T = g * L
    ntrials = 10000

    trajectories_randomised = simulate_ou_trajectories(sigma, mean, tau, dt, T, ntrials, is_randomising=True)
    trajectories = trajectories_randomised
    print("Randomised OU simulation has been completed.")

    hit_times = first_hit_times_zero(trajectories)
    lens = hit_times * dt
    epdf, ecdf, ecdf_conv = calc_epdf_ecdf(lens)

    print("Empirical CDF and PDF have been calculated.")
    return epdf, ecdf, ecdf_conv


def pb_term_bec_lim_iter_min_ecdf_sim_ppd_vn_absorbed(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time, endstar_ppd_vn, gamma_ppd_vn, nu_ppd_vn, theta_ppd_vn, shift_ppd_vn=0.0):
    epdf, ecdf, ecdf_conv = estimate_ou_ecdf_ppd_vn_absorbed(g, gamma_ppd_vn, nu_ppd_vn, theta_ppd_vn, gstar, N, I, init_period, collapse_time, shift_ppd_vn)
    #epdf, ecdf, ecdf_conv = estimate_ou_ecdf_absorbed(g, gamma, nu, theta, gstar, N, L)
    #width = endstar - taustar
    width = 2 * (gamma_ppd_vn * (gstar - g) - shift_ppd_vn) * (endstar_ppd_vn - init_period)
    # width = 2 * gamma_ppd_vn * (gstar - g) * (endstar_ppd_vn - init_period)
    # fer = ecdf_conv(width)
    pdfmin = epdf
    def qfunc(z, x):
        q_a = pdfmin(z)
        q_b = pdfmin(x - z)
        return q_a * q_b
    qfunc_int = lambda x: integrate.quad(qfunc, 0, x, args=(x,))[0]
    def success_func(x):
        return qfunc_int(x)
    success_func_int = integrate.quad(success_func, width, 500)[0]
    fer = 1 - success_func_int

    print(g, fer)
    return fer

def pb_term_bec_lim_iter_min_ecdf_sim(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time):
    amu = mu(g, gamma, nu, theta, gstar, N)
    ecdf, epdf, _ecdf_conv = estimate_ou_ecdf(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time, amu)
    lam = 1 / amu
    width = endstar - taustar
    pdfmin = lambda x: epdf(x) * np.exp(-x * lam) +\
             (1 - ecdf(x)) * lam * np.exp(-x * lam)
    def qfunc(z, x):
        q_a = pdfmin(z)
        q_b = pdfmin(x - z)
        return q_a * q_b
    qfunc_int = lambda x: integrate.quad(qfunc, 0, x, args=(x,))[0]
    def success_func(x):
        return qfunc_int(x)
    success_func_int = integrate.quad(success_func, width, np.inf)[0]
    fer = 1 - success_func_int
    print(g, fer)
    return fer


import scipy.stats as stats
from scipy.stats import norm
import scipy.integrate as integrate

def pb_term_bec_lim_iter_min(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time):
    propagation_iter = I - init_period - collapse_time
    amu = mu(g, gamma, nu, theta, gstar, N)
    lam = 1 / amu
    width = endstar - taustar
    loc = propagation_iter * gamma * (gstar - g)
    scale = np.sqrt(propagation_iter * nu / N) * 5  # from re-fitting
    pdfmin = lambda x: stats.norm.pdf(x, loc=loc, scale=scale) * np.exp(-x * lam) +\
             stats.norm.sf(x, loc=loc, scale=scale) * lam * np.exp(-x * lam)
    def qfunc(z, x):
        q_a = pdfmin(z)
        q_b = pdfmin(x - z)
        return q_a * q_b
    qfunc_int = lambda x: integrate.quad(qfunc, 0, x, args=(x,))[0]
    def success_func(x):
        return qfunc_int(x)
    success_func_int = integrate.quad(success_func, width, np.inf)[0]
    fer = 1 - success_func_int
    print(g, fer)
    return fer


from scipy.stats import norm
import scipy.integrate as integrate
from statsmodels.distributions.empirical_distribution import ECDF

from sklearn.neighbors import KernelDensity


from fl_scaling.scaling_laws_full_BP_unlim_iter import *

def estimate_ou_ecdf(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time, amu=None):
    print(f"Estimating the ECDF and EPDF for epsilon={g}, N={N}, I={I}...")
    if amu is None:
        amu = mu(g, gamma, nu, theta, gstar, N)

    ### PARAMETERS IN LANGEVIN NOTATION
    mean = 0.0

    sigma = np.sqrt(nu / N)
    tau = 1 / theta
    dt = 1 / N
    #T = g * L - taustar
    T = L
    ntrials = 50000
    #ntrials = 100

    threshold = gamma * (gstar - g)
    trajectories_randomised = simulate_ou_trajectories(sigma, mean, tau, dt, T, ntrials, is_randomising=True)
    trajectories = trajectories_randomised
    print("Randomised OU simulation has been completed.")
    
    hs = first_hit_times(trajectories, threshold)

    traj_succ = trajectories[np.where(hs == trajectories.shape[1] - 1)]
    traj_succ = traj_succ[:, :-1]
    traj_succ *= -1
    traj_succ += threshold

    propagation_iter = int(I - init_period - collapse_time)

    lens = np.zeros(traj_succ.shape[0])
    for i, tr in enumerate(traj_succ):
        assert np.all(tr > 0)
        x = 0
        for it in range(propagation_iter):
            x += tr[int(round(x * N))]
        lens[i] = x
    return lens

    # # sample probabilities for a range of outcomes
    # if len(lens) == 0:
    #     ecdf = lambda x: 1
    #     epdf = lambda x: 0
    #     ecdf_conv = lambda x: 1
    # else:
    #     ecdf = ECDF(lens)
        
    #     # fit density
    #     model = KernelDensity(bandwidth=0.1,kernel='gaussian')
    #     sample = lens.reshape((len(lens), 1))
    #     model.fit(sample)

    #     max_lens = np.max(lens)
    #     x_th = np.linspace(0, max_lens, 1000)
    #     values = x_th.reshape((len(x_th), 1))
    #     probabilities = model.score_samples(values)
    #     probabilities = np.exp(probabilities)

    #     prob_conv = np.convolve(probabilities, probabilities)
    #     prob_conv /= np.sum(prob_conv)
    #     x_conv = np.linspace(0, 2*max_lens, 1999)

    #     cdf_conv = np.cumsum(prob_conv)
    #     def ecdf_conv(threshold):
    #         if threshold > 2 * max_lens: return 1
    #         idx = np.argmax(x_conv >= threshold)
    #         cdf_point = cdf_conv[idx]
    #         return cdf_point
    #     def epdf(x_th):
    #         x_th = np.asarray([x_th]) # NB: will fail if an array is passed as a parameter :(
    #         values = x_th.reshape((len(x_th), 1))
    #         probabilities = model.score_samples(values)
    #         probabilities = np.exp(probabilities)
    #         return probabilities

    # print("Empirical CDF and PDF have been calculated.")
    # return ecdf, epdf, ecdf_conv, lens

def estimate_ou_ecdf_zero(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time, amu=None):
    print(f"Estimating the ECDF and EPDF for epsilon={g}, N={N}, I={I}...")
    if amu is None:
        amu = mu(g, gamma, nu, theta, gstar, N)

    ### PARAMETERS IN LANGEVIN NOTATION
    mean = gamma * (gstar - g)

    sigma = np.sqrt(nu / N)
    tau = 1 / theta
    dt = 1 / N
    #T = g * L - taustar
    T = L
    ntrials = 10000 * 5
    #ntrials = 100

    trajectories_randomised = simulate_ou_trajectories(sigma, mean, tau, dt, T, ntrials, is_randomising=True)
    trajectories = trajectories_randomised
    #trajectories_nonrandomised = simulate_ou_trajectories(sigma, mean, tau, dt, T, ntrials, is_randomising=False)
    #trajectories = trajectories_nonrandomised
    print("Randomised OU simulation has been completed.")
    
    hs = first_hit_times_zero(trajectories)

    traj_succ = trajectories[np.where(hs == trajectories.shape[1] - 1)]
    traj_succ = traj_succ[:, :-1]

    propagation_iter = int(round(I - init_period - collapse_time))

    lens = np.zeros(trajectories.shape[0])
    for i, tr in enumerate(trajectories):
        hit_time = hs[i]
        x = 0
        for it in range(propagation_iter):
            cur_t = int(round(x * N))
            if cur_t >= hit_time:
                x = hit_time / N
                break
            x += tr[cur_t]
        lens[i] = x

    # sample probabilities for a range of outcomes
    if len(lens) == 0:
        ecdf = lambda x: 1
        epdf = lambda x: 0
        ecdf_conv = lambda x: 1
    else:
        ecdf = ECDF(lens)
        
        # fit density
        model = KernelDensity(bandwidth=0.1,kernel='gaussian')
        sample = lens.reshape((len(lens), 1))
        model.fit(sample)

        max_lens = np.max(lens)
        x_th = np.linspace(0, max_lens, 1000)
        values = x_th.reshape((len(x_th), 1))
        probabilities = model.score_samples(values)
        probabilities = np.exp(probabilities)

        prob_conv = np.convolve(probabilities, probabilities)
        prob_conv /= np.sum(prob_conv)
        x_conv = np.linspace(0, 2*max_lens, 1999)

        cdf_conv = np.cumsum(prob_conv)
        def ecdf_conv(threshold):
            if threshold > 2 * max_lens: return 1
            idx = np.argmax(x_conv >= threshold)
            cdf_point = cdf_conv[idx]
            return cdf_point
        def epdf(x_th):
            x_th = np.asarray([x_th]) # NB: will fail if an array is passed as a parameter :(
            values = x_th.reshape((len(x_th), 1))
            probabilities = model.score_samples(values)
            probabilities = np.exp(probabilities)
            return probabilities

    print("Empirical CDF and PDF have been calculated.")
    return ecdf, epdf, ecdf_conv



def pb_term_bec_lim_iter_speed_smoothed_ecdf(g, gamma, nu, theta, gstar, taustar, endstar, N, L, ecdf):
    amu = mu(g, gamma, nu, theta, gstar, N)
    lam = 1 / amu
    width = endstar - taustar
    def qfunc(z, x):
        #scale = 1e-3  # for testing
        b = min(z, x - z, width / 2)
        a = width - b
        q_a = 1 - ecdf(a)
        q_b = 1 - ecdf(b)
        return q_a * q_b
    qfunc_int = lambda x: integrate.quad(qfunc, 0, x, args=(x,))[0]
    def success_func(x):
        return lam**2 * np.exp(-lam*x) * qfunc_int(x)
    success_func_int = integrate.quad(success_func, width, np.inf)[0]
    fer = 1 - success_func_int
    print(g, fer)
    return fer


def pb_term_bec_lim_iter_speed_smoothed_ecdf_sim(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time):
    amu = mu(g, gamma, nu, theta, gstar, N)
    ecdf, _epdf, _ecdf_conv = estimate_ou_ecdf(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time, amu)
    lam = 1 / amu
    width = endstar - taustar
    def qfunc(z, x):
        #scale = 1e-3  # for testing
        b = min(z, x - z, width / 2)
        a = width - b
        q_a = 1 - ecdf(a)
        q_b = 1 - ecdf(b)
        return q_a * q_b
    qfunc_int = lambda x: integrate.quad(qfunc, 0, x, args=(x,))[0]
    def success_func(x):
        return lam**2 * np.exp(-lam*x) * qfunc_int(x)
    success_func_int = integrate.quad(success_func, width, np.inf)[0]
    fer = 1 - success_func_int
    print(g, fer)
    return fer

def pb_term_bec_lim_iter_ecdf_conv_sim(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time):
    amu = mu(g, gamma, nu, theta, gstar, N)
    _ecdf, epdf, ecdf_conv = estimate_ou_ecdf(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, collapse_time, amu)
    lam = 1 / amu
    width = endstar - taustar
    # def qfunc(z, x):
    #     a = z
    #     b = x - z
    #     q_a = epdf(a)
    #     q_b = epdf(b)
    #     return q_a * q_b
    # qfunc_int = lambda x: integrate.quad(qfunc, 0, x, args=(x,))[0]
    # def success_func(x):
    #     return qfunc_int(x)
    # success_func_int = integrate.quad(success_func, width, np.inf)[0]
    # fer = 1 - success_func_int
    fer = ecdf_conv(width)
    print(g, fer)
    return fer

def pb_term_bec_lim_iter_speed(g, gamma, nu, theta, gstar, taustar, endstar, N, L, I, init_period, wave_speed, collapse_time):
    amu = mu(g, gamma, nu, theta, gstar, N) / gamma / (gstar - g)
    #wave_speed = L / (endstar - taustar) * gamma * (gstar - g)

    # An alternative way of doing this... Is it any better?
    # Well, it doesn't artificially change the speed of the wave.
    # It uses instead the actual speed of the wave, which may be
    # easier to compute. The problem is that we need (endstar - taustar)
    # and gamma * (gstar - g) in any case!
    # width_boundaries = 0.5 * (L + 4 - wave_speed / gamma / (gstar - g) * (endstar - taustar))
    # L += 4  # dv - 1
    # L -= 2 * width_boundaries

    L = wave_speed / gamma / (gstar - g) * (endstar - taustar)
    d_min = calc_min_propagation_distance(L, I, init_period, wave_speed, collapse_time)
    if d_min > L/2: return 1  # Perhaps too pessimistic? Maybe shift init_period?
    return 1 - (1 + (L - 2*d_min)/amu/wave_speed) * np.exp(-L/amu/wave_speed)

def calc_min_propagation_distance(L, I, init_period, wave_speed, collapse_time):
    propagation_iter = I - init_period - collapse_time
    max_propagation_distance = propagation_iter * wave_speed
    min_propagation_distance = L - max_propagation_distance
    if min_propagation_distance < 0:
        min_propagation_distance = 0
    return min_propagation_distance

# Calculating taustar, endstar for other L from the values for L=50
def num_vns_mid(eps):
    return eps - vss_interp_ldpc_term(eps)**(-1)

def taustar_calc_term(eps, L):
    return taustar_interp_ldpc_term(eps) + (L - 50) * num_vns_mid(eps)

def taustar_calc_nonterm(eps, L):
    return taustar_interp_ldpc(eps) + (L - 50) * num_vns_mid(eps)

def endstar_calc_term(eps, L):
    return eps * L - (eps * 50 - endstar_interp_ldpc_term(eps))

def endstar_calc_nonterm(eps, L):
    return eps * L - (eps * 50 - endstar_interp_ldpc(eps))
