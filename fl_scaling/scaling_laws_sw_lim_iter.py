# Scaling laws for sliding window decoding with limited iterations
#
# Described in
# Roman Sokolovskii, Alexandre Graell i Amat, Fredrik Brännström,
# "Finite-Length Scaling of SC-LDPC Codes With a Limited Number of Decoding Iterations"
# https://arxiv.org/abs/2203.08880
#
# Implemented by Roman Sokolovskii

from scipy import stats

from fl_scaling.ornstein_uhlenbeck import simulate_ou_trajectories, first_hit_times_zero, first_hit_times
from fl_scaling.scaling_laws_full_BP_unlim_iter import *
from fl_scaling.fokker_planck import sim_fokker_planck_iou_sc_ldpc


def calc_w_eff_lim_iter(W, Ipos, init_period_and_collapse_time, wave_speed):
    # while the wave on the right is forming, the window is still moving
    # similarly, while the two wave collapse, the window is also moving
    W_init = W - init_period_and_collapse_time / Ipos
    # once the wave is formed, the window and the wave move towards each other
    speed_eff = wave_speed + 1 / Ipos
    # then it takes meet_time iterations for the wave to meet the left window boundary
    meet_time = W_init / speed_eff
    # in that time, the wave has propagated by w_eff positions
    w_eff = meet_time * wave_speed
    return w_eff


def sim_ou_trajectories_pd(g, gamma, nu, theta, gstar, N, L, amu=None, ntrials=10000):
    if amu is None:
        amu = mu(g, gamma, nu, theta, gstar, N)

    mean = 0.0

    sigma = np.sqrt(nu / N)
    tau = 1 / theta
    dt = 1 / N
    T = L

    threshold = gamma * (gstar - g)
    trajectories_randomised = simulate_ou_trajectories(sigma, mean, tau, dt, T, ntrials, is_randomising=True)
    trajectories = trajectories_randomised
    print("Randomised OU simulation has been completed.")

    hs = first_hit_times(trajectories, threshold)

    #traj_succ = trajectories[np.where(hs == trajectories.shape[1] - 1)]
    trajectories = trajectories[:, :-1]
    trajectories *= -1
    trajectories += threshold

    return trajectories, hs


def sim_ou_trajectories_ppd(g, gamma, nu, theta, gstar, N, num_iter, shift_ppd_vn=0.0, loc = None, ntrials=10000):
    if loc is None:
        mean = gamma * (gstar - g) - shift_ppd_vn
    else:
        mean = loc

    sigma = np.sqrt(nu / N)
    tau = 1 / theta / (gstar - g)
    #dt = 1 / 10
    dt = 1
    T = num_iter

    trajectories_randomised = simulate_ou_trajectories(sigma, mean, tau, dt, T, ntrials, is_randomising=True)
    trajectories = trajectories_randomised
    #print("Randomised OU simulation has been completed.")

    hit_times = first_hit_times_zero(trajectories)
    return trajectories, hit_times

def hesse_iou_fpt_approx_1(t, g, gamma, nu, theta, gstar, N, Iit, Iinit, wave_speed, shift_ppd_vn=0.0, loc = None):
    if loc is None:
        mean = gamma * (gstar - g) - shift_ppd_vn
    else:
        mean = loc

    sigma = np.sqrt(nu / N)
    tau = 1 / theta / (gstar - g)

    hesse_beta = 1 / tau
    hesse_sigma = sigma * np.sqrt(2. * hesse_beta) * wave_speed
    hesse_nu = 0.0
    hesse_mu = -(mean * wave_speed - 1 / Iit)
    # the barrier. 1 - Iinit / Iit is where the linear barrier hits t=0,
    # but the trajectories also start at pos 1, so we need to subtract one
    # from the barrier to make it lower.
    hesse_x = -(- Iinit / Iit)

    g1_t = hesse_iou_fpt_approx_1__(t, hesse_x, hesse_nu, hesse_mu, hesse_beta, hesse_sigma)
    return g1_t

def hesse_iou_fpt_approx_2(t, g, gamma, nu, theta, gstar, N, Iit, Iinit, wave_speed, shift_ppd_vn=0.0, loc = None):
    if loc is None:
        mean = gamma * (gstar - g) - shift_ppd_vn
    else:
        mean = loc

    sigma = np.sqrt(nu / N)
    tau = 1 / theta / (gstar - g)

    hesse_beta = 1 / tau
    hesse_sigma = sigma * np.sqrt(2. * hesse_beta) * wave_speed
    hesse_nu = 0.0
    hesse_mu = -(mean * wave_speed - 1 / Iit)
    # the barrier. 1 - Iinit / Iit is where the linear barrier hits t=0,
    # but the trajectories also start at pos 1, so we need to subtract one
    # from the barrier to make it lower.
    hesse_x = -(- Iinit / Iit)

    g2_t = hesse_iou_fpt_approx_2__(t, hesse_x, hesse_nu, hesse_mu, hesse_beta, hesse_sigma)
    return g2_t

# t              # first passage time of the IOU
# hesse_x        # loc of the barrier
# hesse_nu       # starting point for the OU process
# hesse_mu       # stationary mean of the OU process
# hesse_beta     # covariance decay of the OU process
# hesse_sigma    # sqrt(variance parameter of the OU process)
def hesse_iou_fpt_approx_1__(t, hesse_x, hesse_nu, hesse_mu, hesse_beta, hesse_sigma):
    loc_t = hesse_nu / hesse_beta * (1 - np.exp(-hesse_beta * t)) + hesse_mu * t
    var_t = hesse_sigma**2 / (2 * hesse_beta**3) * (2*hesse_beta * t + 4*np.exp(-hesse_beta * t) - np.exp(-2*hesse_beta*t) - 3)
    scale_t = np.sqrt(var_t)
    norm_pdf = stats.norm.pdf(hesse_x, loc=loc_t, scale=scale_t)
    g1_t = ((3*hesse_x - (hesse_nu + hesse_mu) * t) / (2*t) - hesse_beta / 8 * (3*(hesse_x - hesse_mu*t) - hesse_nu*t)) * norm_pdf
    return g1_t

def hesse_iou_fpt_approx_2__(t, hesse_x, hesse_nu, hesse_mu, hesse_beta, hesse_sigma):
    lt = hesse_lambda_t(t, hesse_x, hesse_nu, hesse_mu, hesse_beta, hesse_sigma)
    norm_pdf = stats.norm.pdf(lt)
    norm_cdf = stats.norm.cdf(lt)
    g1_t = hesse_iou_fpt_approx_1__(t, hesse_x, hesse_nu, hesse_mu, hesse_beta, hesse_sigma)
    g2_t = g1_t * (norm_cdf + norm_pdf / lt)
    return g2_t

def hesse_lambda_t(t, x, nu, mu, beta, sigma):
    # x = -x
    # mu = -mu
    hesse_lambda = (3*x - (nu+mu)*t - (beta*t/4)*(3*(x - mu*t) - nu*t)) / (sigma*t**(3/2) - sigma*beta*t**(5/2)/8)
    return hesse_lambda

# The trajectories represent positions as a function of BP iterations
# => PPD trajectories should be normalized by N and multiplied by vss_interp_ldpc_term(e)
# To convert from the number of VNs decoded to the position of the wave.
def sim_sw_lim_iter_failure_from_trajectories(Iit, Iinit, numPos, trajectories_ss):
    posInit = 1  # at the beginning of the steady state there are no VNs at pos zero

    num_iter = Iit * (numPos - 1) + Iinit

    failed = 0
    total = 0
    for tr in trajectories_ss:
        total += 1
        pos = posInit
        itr = 0
        isBreak = False
        for i in range(numPos):
            numit = Iinit if i == 0 else Iit
            for it in range(numit):
                if tr[itr] <= 0:
                    total -= 1
                    isBreak = True
                    break
                pos += tr[itr]
                itr += 1
            if isBreak: break
            if pos < i + 1:
                failed += 1
                break

    return failed / total, failed, total

def sim_sw_lim_iter_failure_from_trajectories_randomized(Iit, Iinit, Iinit_ones, scale, numPos, trajectories_ss):
    posInit = 1  # at the beginning of the steady state there are no VNs at pos zero

    # # Initial distribution for IOU 
    # one_index = Iinit_ones
    # ones_mean = np.mean(one_index)
    # ones_variance = np.var(one_index)
    # ig_mu = ones_mean
    # ig_lambda = ig_mu ** 3 / ones_variance
    # bm_nu = 1 / ig_mu
    # bm_sigma = np.sqrt(1 / ig_lambda)
    # bm_t = ig_mu
    # bm_loc = bm_nu * bm_t
    # bm_scale = bm_sigma * np.sqrt(bm_t)

    failed = 0
    total = 0
    for tr in trajectories_ss:
        total += 1
        # # Randomization as in Fokker-Planck equation
        # posInit = bm_loc + bm_scale * np.random.randn()
        pos = posInit
        itr = 0
        isBreak = False
        Iinit_ss_rnd = int(Iinit - np.random.choice(Iinit_ones))  # actual thing
        for i in range(numPos):
            #numit = Iinit if i == 0 else Iit # non-randomized
            numit = Iinit_ss_rnd if i == 0 else Iit
            if numit < 0:
                failed += 1
                break
            for it in range(numit):
                if tr[itr] <= 0:
                    total -= 1
                    isBreak = True
                    break
                pos += tr[itr] + np.random.normal(loc = 0, scale = scale)
                # if i > 0:
                #     pos += tr[itr] + np.random.normal(loc = 0, scale = scale)
                # else:
                #     pos += tr[itr]  # Do not randomize at position zero (it has already been done..?)
                itr += 1
            if isBreak: break
            if pos < i + 1:
                failed += 1
                break

    return failed / total, failed, total


def sim_sw_lim_iter_failure_from_ou_pd(g, gamma, nu, theta, gstar, N, L, Iit, Iinit, vn_speed_pd):
    trajectories_ss, hit_times = sim_ou_trajectories_pd(g, gamma, nu, theta, gstar, N, 2 * L)
    numPos = L
    num_iter = Iit * (numPos - 1) + Iinit

    trajectories_ou_pd = []

    for i, tr in enumerate(trajectories_ss):
        x = 0
        abort = False
        traj_ou_pd = np.zeros(num_iter)
        for it in range(num_iter):
            if int(round(x * N)) >= hit_times[i]:
                abort = True
                break
            else:
                cur_val = tr[int(round(x * N))]
                traj_ou_pd[it] = cur_val * vn_speed_pd
                x += cur_val
        if not abort:
            trajectories_ou_pd.append(traj_ou_pd)

    trajectories_ou_pd = np.array(trajectories_ou_pd)
    print(trajectories_ou_pd.shape)
    fail_prob, failed, total = sim_sw_lim_iter_failure_from_trajectories(Iit, Iinit, numPos, trajectories_ou_pd)
    return fail_prob, failed, total


def sim_sw_lim_iter_failure_from_ou_pd_randomized(g, gamma, nu, theta, gstar, N, L, Iit, Iinit, Iinit_ones, scale, vn_speed_pd, ntrials=10000):
    trajectories_ss, hit_times = sim_ou_trajectories_pd(g, gamma, nu, theta, gstar, N, 2 * L, ntrials=ntrials)
    numPos = L
    num_iter = Iit * (numPos - 1) + Iinit

    trajectories_ou_pd = []

    for i, tr in enumerate(trajectories_ss):
        x = 0
        abort = False
        traj_ou_pd = np.zeros(num_iter)
        for it in range(num_iter):
            if int(round(x * N)) >= hit_times[i]:
                abort = True
                break
            else:
                cur_val = tr[int(round(x * N))]
                traj_ou_pd[it] = cur_val * vn_speed_pd
                x += cur_val
        if not abort:
            trajectories_ou_pd.append(traj_ou_pd)

    trajectories_ou_pd = np.array(trajectories_ou_pd)
    print(trajectories_ou_pd.shape)
    fail_prob, failed, total = sim_sw_lim_iter_failure_from_trajectories_randomized(Iit, Iinit, Iinit_ones, scale, numPos, trajectories_ou_pd)
    return fail_prob, failed, total


def sim_sw_lim_iter_failure_from_ou_ppd(g, gamma, nu, theta, gstar, N, L, Iit, Iinit, vn_speed_pd, shift_ppd_vn=0.0, loc = None, ntrials=10000):
    numPos = L
    num_iter = Iit * (numPos - 1) + Iinit
    trajectories_ss, hit_times = sim_ou_trajectories_ppd(g, gamma, nu, theta, gstar, N, num_iter + 1, shift_ppd_vn, loc, ntrials)

    trajectories_ou_ppd = []

    for i, tr in enumerate(trajectories_ss):
        abort = False
        traj_ou_ppd = np.zeros(num_iter)
        for it in range(num_iter):
            if it >= hit_times[i]:
                abort = True
                break
            else:
                cur_val = tr[it]
                traj_ou_ppd[it] = cur_val * vn_speed_pd
        if not abort:
            trajectories_ou_ppd.append(traj_ou_ppd)

    trajectories_ou_ppd = np.array(trajectories_ou_ppd)
    print(trajectories_ou_ppd.shape)
    fail_prob, failed, total = sim_sw_lim_iter_failure_from_trajectories(Iit, Iinit, numPos, trajectories_ou_ppd)
    return fail_prob, failed, total

def sim_sw_lim_iter_failure_from_ou_ppd_randomized(g, gamma, nu, theta, gstar, N, L, Iit, Iinit, Iinit_ones, scale, vn_speed_pd, shift_ppd_vn=0.0, loc = None, ntrials=10000):
    numPos = L
    num_iter = Iit * (numPos - 1) + Iinit
    trajectories_ss, hit_times = sim_ou_trajectories_ppd(g, gamma, nu, theta, gstar, N, num_iter + 1, shift_ppd_vn, loc, ntrials)

    trajectories_ou_ppd = []

    for i, tr in enumerate(trajectories_ss):
        abort = False
        traj_ou_ppd = np.zeros(num_iter)
        for it in range(num_iter):
            if it >= hit_times[i]:
                abort = True
                break
            else:
                cur_val = tr[it]
                traj_ou_ppd[it] = cur_val * vn_speed_pd
        if not abort:
            trajectories_ou_ppd.append(traj_ou_ppd)

    trajectories_ou_ppd = np.array(trajectories_ou_ppd)
    print(trajectories_ou_ppd.shape)
    fail_prob, failed, total = sim_sw_lim_iter_failure_from_trajectories_randomized(Iit, Iinit, Iinit_ones, scale, numPos, trajectories_ou_ppd)
    return fail_prob, failed, total


def fer_sw_lim_iter_from_sim_ou_pd(e, gamma, nu, theta, gstar, taustarf, taustarf_nonterm, endstarf, N, L, W, Iit, Iinit, taustar_ppd_vn, vn_speed_pd):
    init_period = int(round(float(taustar_ppd_vn)))
    Iinit = Iinit - init_period
    fail_prob = sim_sw_lim_iter_failure_from_ou_pd(e, gamma, nu, theta, gstar, N, L, Iit, Iinit, vn_speed_pd)[0]
    w_eff_ph2 = calc_w_eff_lim_iter(W, Iit, taustar_ppd_vn, vn_speed_pd * gamma * (gstar - e))

    fer_ph2 = pb_term(e, gamma, nu, theta, gstar, taustarf(e,w_eff_ph2), endstarf(e,w_eff_ph2), N, L=w_eff_ph2)
    fer_ph1_unlim = pb(e, gamma, nu, theta, gstar, taustarf_nonterm(e,L - w_eff_ph2), e * (L - w_eff_ph2), N, L=(L - w_eff_ph2))

    fer_overall = 1 - (1 - fail_prob) * (1 - fer_ph1_unlim) * (1 - fer_ph2)
    print(e, fer_overall)
    return fer_overall


def fer_sw_lim_iter_from_sim_ou_pd_randomized(e, gamma, nu, theta, gstar, fer_ph1_unlim, fer_ph2, N, L, W, Iit, Iinit, Iinit_ones, scale, taustar_ppd_vn, vn_speed_pd, ntrials=10000):
    #init_period = int(round(float(taustar_ppd_vn)))
    #Iinit = Iinit - init_period
    fail_prob = sim_sw_lim_iter_failure_from_ou_pd_randomized(e, gamma, nu, theta, gstar, N, L, Iit, Iinit, Iinit_ones, scale, vn_speed_pd, ntrials)[0]

    fer_overall = 1 - (1 - fail_prob) * (1 - fer_ph1_unlim) * (1 - fer_ph2)
    print(e, fer_overall)
    return fer_overall


def fer_sw_lim_iter_from_sim_ou_ppd(e, gamma, nu, theta, gstar, fer_ph1_unlim, fer_ph2, N, L, W, Iit, Iinit, taustar_ppd_vn, vn_speed_pd, shift_ppd_vn=0.0, loc = None, ntrials=10000):
    init_period = int(round(float(taustar_ppd_vn)))
    Iinit = Iinit - init_period
    fail_prob = sim_sw_lim_iter_failure_from_ou_ppd(e, gamma, nu, theta, gstar, N, L, Iit, Iinit, vn_speed_pd, shift_ppd_vn, loc, ntrials)[0]

    fer_overall = 1 - (1 - fail_prob) * (1 - fer_ph1_unlim) * (1 - fer_ph2)
    print(e, fer_overall)
    return fer_overall

def fer_sw_lim_iter_from_sim_ou_ppd_randomized(e, gamma, nu, theta, gstar, fer_ph1_unlim, fer_ph2, N, L, W, Iit, Iinit, Iinit_ones, scale, taustar_ppd_vn, vn_speed_pd, shift_ppd_vn=0.0, loc = None, ntrials=10000):
    #init_period = int(round(float(taustar_ppd_vn)))
    #Iinit = Iinit - init_period
    fail_prob = sim_sw_lim_iter_failure_from_ou_ppd_randomized(e, gamma, nu, theta, gstar, N, L, Iit, Iinit, Iinit_ones, scale, vn_speed_pd, shift_ppd_vn, loc, ntrials)[0]

    fer_overall = 1 - (1 - fail_prob) * (1 - fer_ph1_unlim) * (1 - fer_ph2)
    print(e, fer_overall)
    return fer_overall

#def fer_sw_lim_iter_from_fokker_planck_ou_ppd(e, gamma, nu, theta, gstar, fer_ph1_unlim, fer_ph2, N, L, W, Iit, Iinit, Iinit_ones, scale, taustar_ppd_vn, vn_speed_pd, shift_ppd_vn=0.0, loc = None, dt = 1, rho=0.0, therange=None):
def fer_sw_lim_iter_from_fokker_planck_ou_ppd(e, gamma, nu, theta, gstar, fer_ph1_unlim, fer_ph2, N, L, W, Iit, Iinit, scale, taustar_ppd_vn, vn_speed_pd, shift_ppd_vn=0.0, loc = None, dt = 1, rho=0.0, therange=None):
    if loc is None:
        mean_ou = gamma * (gstar - g) - shift_ppd_vn
    else:
        mean_ou = loc
    #fail_prob = sim_fokker_planck_iou_sc_ldpc(e, mean_ou, nu, theta, gstar, N, L, W, Iit, Iinit, Iinit_ones, scale, taustar_ppd_vn, vn_speed_pd, dt, rho, therange)[0]
    fail_prob = sim_fokker_planck_iou_sc_ldpc(e, mean_ou, nu, theta, gstar, N, L, W, Iit, Iinit, scale, taustar_ppd_vn, vn_speed_pd, dt, rho, therange)[0]

    fer_overall = 1 - (1 - fail_prob) * (1 - fer_ph1_unlim) * (1 - fer_ph2)
    print(e, fer_overall)
    return fer_overall
