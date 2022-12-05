## Density evolution for SC-LDPC codes, frame-Asyncronous CSA and IRSA
import numpy as np
from numpy.polynomial.polynomial import polyval, polyder
import scipy.io
from scipy.stats import binom
from scipy.special import comb

import pickle

import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)

def de_fa_uniform(vnupd, cnupd, endupd, frame_len):
    de_span = frame_len * 20
    max_iter = 100000
    threshold = 1e-20
    delta_threshold = 1e-100
    p = np.ones(de_span)
    q = np.ones(de_span)

    prev_mean = 0
    for it in range(max_iter):
        for i in range(de_span):
            p[i] = vnupd(q_mean(q, i, frame_len), i)
        for i in range(de_span):
            q[i] = cnupd(p_mean(p, i, frame_len), i)
        cur_mean = np.mean(p)
        if np.all(p[0 : de_span] < threshold):
            print("threshold break")
            break
        if np.abs(cur_mean - prev_mean) < delta_threshold:
            #pass
            print("fixpoint break")
            break
        prev_mean = cur_mean
        if it % 500 == 0 and it != 0:
            print("iter mean:", cur_mean)

    plrs = [endupd(q_mean(q, i, frame_len)) for i in range(de_span)]
    plr = np.mean(plrs)
    return plr

def de_fa_irsa_uniform(g, vn_deg, vn_pr, frame_len):
    Lambda = vn_spec_to_Lambda(vn_deg, vn_pr)
    Lambda_der = polyder(Lambda)
    Lambda_der_1 = polyval(1.0, Lambda_der)

    lam = Lambda_der / Lambda_der_1
    vnupd = lambda q_m, i: polyval(q_m, lam)
    cnupd = lambda p_m, i: 1 - np.exp(-1 * prev_window(i, frame_len) * g / frame_len * Lambda_der_1 * p_m)
    endupd = lambda q_m: polyval(q_m, Lambda)
    print("trying g:", g)
    plr = de_fa_uniform(vnupd, cnupd, endupd, frame_len)
    print("ret:", g, plr)
    return plr

# n - total number of packets that a user will transmit
# k - the number of packets that will be enough to recover user's message
# transmission of an MDS (Maximum Distance Separable) code is therefore assumed.
# Lambda_der_1 - in IRSA, this is the average degree of VN node.
# In CSA with fixed code, the degree of VN node is fixed and is equal to n.
def de_fa_csa_uniform(g, n, k, frame_len):
    vnupd = lambda q_m, i: pr_n_k_mds_erasure(n, k, q_m)
    cnupd = lambda p_m, i: 1 - np.exp(-1 * prev_window(i, frame_len) * g / frame_len * n / k * p_m)
    endupd = lambda q_m: q_m ** n
    print("trying g:", g)
    plr = de_fa_uniform(vnupd, cnupd, endupd, frame_len)
    print("RET", g, plr)
    return plr

def q_mean(q, i, frame_len):
    return np.mean(q[i : i + frame_len])

def p_mean(p, i, frame_len):
    window = prev_window(i, frame_len)
    return np.mean(p[i - window + 1 : i + 1])


b = {}
for n in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]:
    b[n] = {}
    for k in range(n):
        b[n][k] = comb(n, k)


def pr_n_k_mds_erasure(n, k, p_erasure):
    p_success = 1 - p_erasure
    num_incoming = n - 1
    max_successes = k - 1 # max successes to still have a failure
    p_erasure_out = sum([b[num_incoming][nsuc] * p_success ** nsuc * (1 - p_success) ** (num_incoming - nsuc) for nsuc in range(k) ])
    #p_erasure_out = binom.cdf(num_incoming, max_successes, p_success)
    #print(p_erasure_out)
    return p_erasure_out

def de_fa_irsa_fixed(vn_deg, vn_pr, frame_len):
    pass

# number of slots before and including the current one
def prev_window(i, frame_len):
    return frame_len if i >= frame_len - 1 else i + 1

def vn_spec_to_Lambda(deg, pr):
    Lambda = np.zeros(deg[-1] + 1)
    for i in range(len(deg)):
        Lambda[deg[i]] = pr[i]
    return Lambda

def estimate_wave_pos(plrs):
    max_plr = np.max(plrs)
    if max_plr < 1e-8:
        return -1
    wave_lvl = max_plr / 2
    wave_pos = np.argmax(plrs > wave_lvl)  # The index of the first True is returned
    return wave_pos

def find_wave_indices(wave_positions, min_pos, max_pos):
    min_index = np.argmax(wave_positions >= min_pos)
    max_index = np.argmax(wave_positions >= max_pos)
    return min_index, max_index

def estimate_wave_speed(wave_positions, speed_period, ld):
    min_pos = ld + 2
    max_pos = np.max(wave_positions)
    min_index, max_index = find_wave_indices(wave_positions, min_pos, max_pos)

    iters = (max_index - min_index) * speed_period / (max_pos - min_pos)
    wave_speed = 1 / iters
    return wave_speed

def estimate_wave_width(plrs, init_period_threshold, mid_ss_level):
    # NB: Assumes that we are in the steady state
    min_index_left = np.argmax(plrs >= init_period_threshold)
    max_index_left = np.argmax(plrs >= mid_ss_level * 0.95)
    wave_width_left = max_index_left - min_index_left
    plrs_flipped = plrs[::-1]
    min_index_right = np.argmax(plrs_flipped >= init_period_threshold)
    max_index_right = np.argmax(plrs_flipped >= mid_ss_level * 0.95)
    wave_width_right = max_index_right - min_index_right
    assert wave_width_left == wave_width_right  # Test
    wave_width = (wave_width_left + wave_width_right) / 2
    return wave_width

def de_sc_ldpc(e, ld, rd, L):
    print(f"Trying e={e}")
    max_iter = int(1e6)
    speed_period = 1
    threshold = 1e-20
    delta_threshold = 1e-100

    init_period_threshold = 1e-8  # v1
    init_period_threshold = 7.875503501162272e-05  # v2
    init_period = -1
    waves_established = False

    wave_positions = []
    wave_widths = []

    mid_ss_level = 0
    collapse_begun = False
    collapse_it = 0

    cn_span = L+ld-1
    p = np.ones((ld, cn_span))  # p[i,j] denotes the probability that the message from a VN at position j-i to a CN at position j is an erasure
    q = np.ones((ld, L))        # q[i,j] denotes the probability that the message from a CN at position j+i to a VN at position j is an erasure

    perr_cn_mean = np.ones(cn_span) # perr_cn_mean[i] denotes the probability that the message from a CN at position i is an erasure

    #plrs_all = np.zeros((max_iter, L))
    tot_plr_prev = e * L
    delta_plrs = []

    prev_mean = 0
    for it in range(max_iter):
        for j in range(cn_span):
            for i in range(ld):
                vnpos = j-i
                cnpos = j
                if vnpos < 0 or vnpos >= L:
                    p[i,j] = 0  # All VNs outside of the chain boundaries are resolved
                    continue
                p[i,j] = e * np.prod(q[0:i,vnpos]) * np.prod(q[i+1:,vnpos]) #  / q[i,vnpos] Creates dividing by zero problems

        for j in range(L):
            for i in range(ld):
                vnpos = j
                cnpos = j+i
                if cnpos > cn_span: continue
                # inc = int(rd / ld)  # if this is not integer, I'm in trouble
                # q[i,j] = 1 - np.prod(1-p[:,cnpos])**inc / (1-p[i,cnpos])
                perr_mean = np.mean(p[:,cnpos])
                inc = rd - 1
                q[i,j] = 1 - (1 - perr_mean)**inc
                perr_cn_mean[cnpos] = perr_mean

        #cur_mean = np.mean(p)
        plrs = np.array([e * np.prod(q[:,vnpos]) for vnpos in range(L)])
        tot_plr = np.sum(plrs)
        delta_plr = tot_plr_prev - tot_plr
        tot_plr_prev = tot_plr
        delta_plrs.append(delta_plr)
        # plt.title(str(it))
        # plt.plot(plrs)
        # plt.ylim(bottom=0)
        # plt.xlim(left=0,right=49)
        # plt.show()
        #plrs_all[it, :] = plrs
        cur_mean = np.mean(plrs)
        if not waves_established and plrs[0] < init_period_threshold and plrs[-1] < init_period_threshold:
            init_period = it + 1  # the iteration counter is zero-based
            mid_ss_level = plrs[int(len(plrs) / 2)]
            waves_established = True
        mid_level = plrs[int(len(plrs) / 2)]
        mid_level_cn = perr_cn_mean[int(len(perr_cn_mean) / 2)]
        if not collapse_begun and mid_level < 0.99 * mid_ss_level:
            collapse_it = it
            collapse_begun = True
        if waves_established and not collapse_begun:  # <=> if we are in the steady state
            wave_widths.append(estimate_wave_width(perr_cn_mean, init_period_threshold, mid_level_cn))
        if it % speed_period == 0 and it != 0:
            wave_pos = estimate_wave_pos(plrs)
            wave_positions.append(wave_pos)
        if np.all(p[:,0 : cn_span] < threshold):
            print("threshold break")
            break
        if np.abs(cur_mean - prev_mean) < delta_threshold:
            print("fixpoint break")
            break
        prev_mean = cur_mean
        #if it == 100 - 1: break
        if it % 500 == 0 and it != 0:
            print("iter mean:", cur_mean)

    left_wave_beginning = np.argmax(plrs > init_period_threshold)
    print(f"LEFT BEG {it} iterations: {left_wave_beginning}")
    wave_positions = np.array(wave_positions)
    wave_speed = estimate_wave_speed(wave_positions, speed_period, ld)
    wave_width = np.mean(wave_widths)
    print(wave_widths)

    if collapse_begun:
        collapse_time = it - collapse_it
    else:
        collapse_time = 0

    # with open('taustar_endstar_collapse_ldpc_5_10_terminated_ppd_vn.dat', 'rb') as f:
    #     gs, tstars, endstars, collapse_times = pickle.load(f)
    # for x in zip(gs, tstars, endstars, collapse_times):
    #     print(x)
    # from scipy.interpolate import interp1d
    # taustar_interp_ldpc_term_ppd_vn = interp1d(gs, tstars)
    # endstar_interp_ldpc_term_ppd_vn = interp1d(gs, endstars)
    # collapse_interp_ldpc_term_ppd_vn = interp1d(gs, collapse_times)
    # plt.plot(delta_plrs)
    # plt.axvline(x=init_period)
    # plt.axvline(x=it - collapse_time)
    # plt.axvline(x=taustar_interp_ldpc_term_ppd_vn(0.47), color='red')
    # plt.axvline(x=endstar_interp_ldpc_term_ppd_vn(0.47), color='red')
    # plt.axvline(x=it - collapse_interp_ldpc_term_ppd_vn(0.47), color='black')
    # plt.xlim(left=0)
    # plt.ylim(bottom=0)
    # plt.show()
    # with open(f'delta_vn_de_sc_ldpc_{ld}_{rd}_{L}_{e}_term.pkl', 'wb') as f:
    #     pickle.dump([delta_plrs, it], f)

    plrs = [e * np.prod(q[:,vnpos]) for vnpos in range(L)]
    plr = np.mean(plrs)
    print(plr)
    #return plr, init_period, wave_speed, collapse_time, wave_width, plrs_all
    return plr, init_period, wave_speed, collapse_time, wave_width, None, delta_plrs


def de_sw_sc_ldpc(e, ld, rd, L, W, I):
    ms = ld - 1
    cn_span = L + ms
    p = np.ones((ld, cn_span))  # p[i,j] denotes the probability that the message from a VN at position j-i to a CN at position j is an erasure
    q = np.ones((ld, L))        # q[i,j] denotes the probability that the message from a CN at position j+i to a VN at position j is an erasure

    perr_cn_mean = np.ones(cn_span) # perr_cn_mean[i] denotes the probability that the message from a CN at position i is an erasure

    e_evol = []

    prev_mean = 0

    # VN init
    for j in range(cn_span):
        for i in range(ld):
            vnpos = j-i
            cnpos = j
            if vnpos < 0 or vnpos >= L:
                p[i,j] = 0  # All VNs outside of the chain boundaries are resolved
                continue
            p[i,j] = e

    for posW in range(cn_span):
        startCN = posW
        endCN = startCN + W
        if endCN > cn_span: endCN = cn_span
        if posW <= ms:
            startVN = 0
            endVN = W + posW
        else:
            startVN = posW - ms
            endVN = startVN + W + ms
        if endVN > L: endVN = L

        theI = 100 if posW == 0 else I
        # theI = I  # The orthodox way
        for it in range(theI):
            # CN update
            for cnpos in range(startCN, endCN):
                for i in range(ld):
                    vnpos = cnpos - i
                    if vnpos < 0 or vnpos >= L: continue
                    perr_mean = np.mean(p[:,cnpos])
                    inc = rd - 1
                    q[i,vnpos] = 1 - (1 - perr_mean)**inc
                    perr_cn_mean[cnpos] = perr_mean

            # VN update
            for vnpos in range(startVN, endVN):
                for i in range(ld):
                    cnpos = vnpos + i
                    if vnpos < 0 or vnpos >= L:
                        p[i,cnpos] = 0  # All VNs outside of the chain boundaries are resolved
                        continue
                    p[i,cnpos] = e * np.prod(q[0:i,vnpos]) * np.prod(q[i+1:,vnpos]) #  / q[i,vnpos] Creates dividing by zero problems

            plrs = np.array([e * np.prod(q[:,vnpos]) for vnpos in range(L)])
            e_evol.append((plrs, startVN, endVN, startCN, endCN))

    return e_evol


def gen_r_sc_ldpc(e, l, r, m, frame_len, total_size, is_terminated):
    max_degree = r
    ra = np.zeros((total_size, max_degree + 2))
    for u in range(total_size):
        lu = prev_window(u, frame_len)
        for j in range(1, max_degree + 1):
            agg = 0.0
            for d in range(j, max_degree + 1):
                if lu < frame_len:
                    # l * m edges goes out of VNs at each position
                    # l * m edges goes into CNs at each position,
                    # but incoming edges are uniformly distributed among
                    # frame_len positions. As such, each position is responsible
                    # for l * m / frame_len edges coming into CNs at current position.
                    # So, if there are only lu < frame_len positions connected to
                    # CNs at current position, only lu * l * m / frame_len CN slots
                    # out of total l * m slots will be occupied. So the probability
                    # of a CN slot to be occupied is:
                    # lu * l * m / frame_len / l / m = lu / frame_len
                    # Surprisingly enough, this probability does not depend on l.
                    pdu = binom.pmf(d, r, lu / frame_len)
                else:
                    pdu = 1 if d == r else 0
                agg += pdu * binom.pmf(j, d, e)
            ra[u, j] = j * m * l / r * agg
    if is_terminated:
        ra[-1:-(frame_len + 1):-1, :] = ra[0:frame_len, :]
    return ra


def de_sc_ldpc_doping(e, ld, rd, L, doping_points, quit_threshold):
    print(f"Trying e={e}")
    max_iter = int(1e6)
    threshold = 1e-20
    delta_threshold = 1e-100
    cn_span = L+ld-1
    #vns_from, vns_to = 2 * ld, min(doping_points) - 2 * ld + 1  # PLR should be evaluated in that region, the tails are useless
    vns_from, vns_to = 3 * ld, L - 3 * ld + 1
    #vns_from, vns_to = 0, L - 1
    p = np.ones((ld, cn_span))  # p[i,j] denotes the probability that the message from a VN at position j-i to a CN at position j is an erasure
    q = np.ones((ld, L))        # q[i,j] denotes the probability that the message from a CN at position j+i to a VN at position j is an erasure

    prev_mean = 0
    for it in range(max_iter):
        for j in range(cn_span):
            for i in range(ld):
                vnpos = j-i
                cnpos = j
                # in DE for doped ensembles, we are not interested in termination
                if vnpos < 0 or vnpos >= L:
                    p[i,j] = 1  # All VNs outside of the chain boundaries are NOT resolved
                    continue
                if vnpos in doping_points:
                    p[i,j] = 0  # All VNs in the doped positions are resolved
                    continue
                p[i,j] = e * np.prod(q[0:i,vnpos]) * np.prod(q[i+1:,vnpos]) #  / q[i,vnpos] Creates dividing by zero problems

        for j in range(L):
            for i in range(ld):
                vnpos = j
                cnpos = j+i
                if cnpos > cn_span: continue
                inc = int(rd / ld)  # if this is not integer, I'm in trouble
                q[i,j] = 1 - np.prod(1-p[:,cnpos])**inc / (1-p[i,cnpos])

        #cur_mean = np.mean(p)
        plrs = [e * np.prod(q[:,vnpos]) for vnpos in range(L)]
        plrs_middle = plrs[vns_from : vns_to]
        cur_mean = np.mean(plrs_middle)
        if np.all(p[:,vns_from : vns_to] < threshold):
            print("threshold break")
            break
        if np.abs(cur_mean - prev_mean) < delta_threshold:
            print(f"fixpoint break, iteration={it}")
            break
        if cur_mean < quit_threshold:
            print("quit_threshold")
            break
        prev_mean = cur_mean
        if it % 500 == 0 and it != 0:
            print("iter mean:", cur_mean)

    plrs = [e * np.prod(q[:,vnpos]) for vnpos in range(L)]
    plrs_middle = plrs[vns_from : vns_to]
    plr = np.mean(plrs_middle)
    print(plr)
    return plr


# DOPING for SMOOTHING at the CNs --- the ensemble we are actually considering
def de_sc_ldpc_doping_semi_structured(e, ld, rd, L, doping_points, quit_threshold):
    print(f"Trying e={e}")
    max_iter = int(1e6)
    threshold = 1e-20
    delta_threshold = 1e-100
    cn_span = L+ld-1
    #vns_from, vns_to = 2 * ld, min(doping_points) - 2 * ld + 1  # PLR should be evaluated in that region, the tails are useless
    #vns_from, vns_to = 3 * ld, L - 3 * ld + 1
    vns_from, vns_to = 0, L
    p = np.ones((ld, cn_span))  # p[i,j] denotes the probability that the message from a VN at position j-i to a CN at position j is an erasure
    q = np.ones((ld, L))        # q[i,j] denotes the probability that the message from a CN at position j+i to a VN at position j is an erasure

    is_soft = isinstance(doping_points, dict)

    prev_mean = 0
    for it in range(max_iter):
        for j in range(cn_span):
            for i in range(ld):
                vnpos = (j-i) % L
                cnpos = j
                # in DE for doped ensembles, we are not interested in termination
                # if vnpos < 0 or vnpos >= L:
                #     p[i,j] = 1  # All VNs outside of the chain boundaries are NOT resolved
                #     continue
                if is_soft:
                    if vnpos in doping_points:
                        alpha = doping_points[vnpos]
                    else:
                        alpha = 0.0
                else:  # HARD doping
                    if vnpos in doping_points:
                        alpha = 1.0
                    else:
                        alpha = 0.0
                # if vnpos in doping_points:
                #     p[i,j] = 0  # All VNs in the doped positions are resolved
                #     continue
                p[i,j] = e * (1.0 - alpha) * np.prod(q[0:i,vnpos]) * np.prod(q[i+1:,vnpos]) #  / q[i,vnpos] Creates dividing by zero problems
                # This is for fully randomized (smoothed) ensemble (as in Camm16!)
                # pinc_mean = np.mean(q[:,vnpos])
                # p[i, j] = e * (1.0 - alpha) * pinc_mean ** (ld - 1)

        for j in range(L):
            for i in range(ld):
                vnpos = j
                cnpos = (j+i) % L
                if cnpos > cn_span: continue
                perr_mean = np.mean(p[:,cnpos])
                inc = rd - 1
                q[i,j] = 1 - (1 - perr_mean)**inc

        #cur_mean = np.mean(p)
        plrs = [e * np.prod(q[:,vnpos]) for vnpos in range(L)]
        plrs_middle = plrs[vns_from : vns_to]
        cur_mean = np.mean(plrs_middle)
        if np.all(p[:,vns_from : vns_to] < threshold):
            print("threshold break")
            break
        if np.abs(cur_mean - prev_mean) < delta_threshold:
            print(f"fixpoint break, iteration={it}")
            break
        if cur_mean < quit_threshold:
            print("quit_threshold")
            break
        prev_mean = cur_mean
        if it % 500 == 0 and it != 0:
            print("iter mean:", cur_mean)

    plrs = [e * np.prod(q[:,vnpos]) for vnpos in range(L)]
    plrs_middle = plrs[vns_from : vns_to]
    plr = np.mean(plrs_middle)
    print(plr)
    return plr


# Binary search for a monotonic function.
# Given that testfun is defined on a range from low to high and
# is false from low to some point, then it is true from this point
# to high, return a range from low to high where the distance from
# low to high is smaller than delta, for which low is still false
# and high is already true.
def binsearch(testfun, frm, to, precision):
    low = frm
    high = to
    while high - low > precision:
        middle = low + (high - low) / 2
        if testfun(middle):
            high = middle
        else:
            low = middle
    return low, high

def fa_irsa_threshold_load(vn_deg, vn_pr, frame_len):
    de_fun = lambda g: de_fa_irsa_uniform(g, vn_deg, vn_pr, frame_len) > 1e-10
    g_star, _ = binsearch(de_fun, 0.0, 1.0, 1e-3)
    return g_star

def fa_csa_threshold_load(n, k, frame_len):
    de_fun = lambda g: de_fa_csa_uniform(g, n, k, frame_len) > 1e-20
    g_star, _ = binsearch(de_fun, 0.0, 1.0, 1e-3)
    return g_star

def sc_ldpc_threshold_load(ld, rd, L):
    de_fun = lambda e: de_sc_ldpc(e, ld, rd, L)[0] > 1e-20
    e_star_low, e_star_high = binsearch(de_fun, 0.0, 0.5, 1e-5)
    return e_star_low, e_star_high

def sc_ldpc_threshold_load_doping(ld, rd, L, doping_points, is_proto=False):
    quit_threshold = 1e-2
    if is_proto:
        de_fun = lambda e: de_sc_ldpc_doping(e, ld, rd, L, doping_points, quit_threshold) > quit_threshold
    else:
        de_fun = lambda e: de_sc_ldpc_doping_semi_structured(e, ld, rd, L, doping_points, quit_threshold) > quit_threshold
    e_star_low, e_star_high = binsearch(de_fun, 0.0, 0.5, 1e-4)
    return e_star_low, e_star_high

def test():
    vn_deg = [3, 8]
    vn_pr = [0.86, 0.14]
    Lambda = vn_spec_to_Lambda(vn_deg, vn_pr)
    Lambda_der = polyder(Lambda)
    lam = Lambda_der / polyval(1.0, Lambda_der)
    print(Lambda)
    print(polyval(1.0, Lambda))
    print(polyder(Lambda))
    print(lam)
    print(q_mean(np.array([1,2,3,4]), 1, 2))
    testfun = lambda x: x > 1.4e-5
    g_star, _ = binsearch(testfun, 0, 1.0, 1e-10)
    print(g_star)

def test_de():
    vn_deg = [3, 8]
    vn_pr = [0.86, 0.14]
    frame_len = 100
    g = 0.963
    plr = de_fa_irsa_uniform(g, vn_deg, vn_pr, frame_len)
    #print('{:<2} {:>12.8f}'.format(g, plr))
    print(plr)

# XXX: ignore the fact that we are dividing frame into slices
def de_csa():
    frame_len = 100
    #ks = [3,4,6,20]
    ks = [6]
    for k in ks:
        fname = "gstars_csa_" + str(k)
        rname = "rates_" + str(k)
        nname = "ns_" + str(k)
        gname = "gstars_" + str(k)
        i = 0
        ns = np.arange(k + 1, k + 1 + 7)
        rates = np.zeros(len(ns))
        gstars = np.zeros(len(ns))
        for n in ns:
            r = k / n
            print("testing k, n, r:", k, n, r)
            gstar = fa_csa_threshold_load(n, k, frame_len)
            print("result: k, n, gstar:", k, n, gstar)
            rates[i] = r
            gstars[i] = gstar
            i += 1
        #scipy.io.savemat(fname, {rname : rates, nname : ns, gname : gstars})

def de_irsa():
    #vn_deg = [3, 8]
    #vn_pr = [0.86, 0.14]
    vn_mono_pr = [1.0]
    frame_len = 100
    degs = [3, 4, 5, 6, 7, 8]
    gstars = np.zeros(len(degs))
    for i in range(len(degs)):
        deg = degs[i]
        g_star = fa_irsa_threshold_load([deg], vn_mono_pr, frame_len)
        gstars[i] = g_star
        print("DEG:", deg, "GSTAR:", g_star)
    #g_star = fa_irsa_threshold_load(vn_deg, vn_pr, frame_len)
    #print("#GSTAR:", g_star)
    scipy.io.savemat('gstars', { 'degs' : degs, 'gstars' : gstars })

def test_de_sc_ldpc():
    ld = 4
    rd = 8
    is_proto = False
    # When we have doping, we look in the middle of the chain and we have truncations
    #L = 50
    #doping_points = {}
    L = 100
    #doping_points = { 49 : 0.75, 50 : 0.2, 51 : 0.75, 52 : 0.2, 53 : 0.75 }
    doping_points = {50, 52}
    # L = 20
    # doping_points = { 10 : 0.324, 11 : 0.149, 12 : 0.350 }   # from the paper on tail-biting (optimized for e = 0.48)
    # This is for doping on the edges, a more smooth termination. No gain in the threshold.
    # But we could have it the other way around, perhaps... And have a smaller rate loss at the boundaries instead!
    # doping_points = { 1, 3, 5, 96, 98, 100 }
    L += len(doping_points)

    # center_pos = 50
    # with open("thresholds_doped_spaced_d4_sc_ldpc_6_12_semi.dat", "wt") as f:
    #     for da in range(4):
    #         for db in range(4):
    #             for de in range(4):
    #                 doping_points = { center_pos - da - 1, center_pos, center_pos + db + 1, center_pos + db + 1 + de + 1 }
    #                 L = 100 + len(doping_points)
    #                 print(da, db, de, doping_points)
    #                 e_star_low, e_star_high = sc_ldpc_threshold_load_doping(ld, rd, L, doping_points, is_proto)
    #                 print(f"({ld},{rd},L={L},doping_points={doping_points}) SC-LDPC code threshold estar: {e_star_low} - {e_star_high}")
    #                 print(da, db, de, e_star_low, e_star_high, file=f)

    # if len(doping_points) == 0:
    #     e_star_low, e_star_high = sc_ldpc_threshold_load(ld, rd, L)
    #     print(f"({ld},{rd},L={L}) SC-LDPC code threshold estar: {e_star_low} - {e_star_high}")
    # else:
    #     e_star_low, e_star_high = sc_ldpc_threshold_load_doping(ld, rd, L, doping_points, is_proto)
    #     print(f"({ld},{rd},L={L},doping_points={doping_points}) SC-LDPC code threshold estar: {e_star_low} - {e_star_high}")
    # with open("thresholds_doped_sc_ldpc_5_10_semi_tailbining.dat", "wt") as f:
    #     for num_dopings in range(ld):
    #         doping_points = { i for i in range(25, 25 + num_dopings) }
    #         L = 50 + len(doping_points)
    #         e_star_low, e_star_high = sc_ldpc_threshold_load_doping(ld, rd, L, doping_points, is_proto)
    #         print(f"({ld},{rd},L={L},doping_points={doping_points}) SC-LDPC code threshold estar: {e_star_low} - {e_star_high}")
    #         print(num_dopings, e_star_low, file=f)
    # with open("thresholds_doped_spaced_d2_sc_ldpc_5_15_proto.dat", "wt") as f:
    #     for distance in range(3 * ld + 1):
    #         doping_points = { 50, 51 + distance }
    #         L = 100 + len(doping_points)
    #         e_star_low, e_star_high = sc_ldpc_threshold_load_doping(ld, rd, L, doping_points, is_proto)
    #         print(f"({ld},{rd},L={L},doping_points={doping_points}) SC-LDPC code threshold estar: {e_star_low} - {e_star_high}")
    #         print(distance, e_star_low, e_star_high, file=f)

    e_star_low, e_star_high = sc_ldpc_threshold_load_doping(ld, rd, L, doping_points, is_proto)
    print(f"({ld},{rd},L={L},doping_points={doping_points}) SC-LDPC code threshold estar: {e_star_low} - {e_star_high}")


def estimate_de_speed():
    ld = 5
    rd = 10
    L = 50
    #es = np.linspace(0.455, 0.49, 7 * 8 + 1)
    es = np.array([0.47])
    init_periods = np.zeros(es.shape)
    speeds = np.zeros(es.shape)
    collapse_times = np.zeros(es.shape)
    wave_widths = np.zeros(es.shape)
    i = 0
    for e in es:
        plr, init_period, wave_speed, collapse_time, wave_width, plrs_all, delta_plrs = de_sc_ldpc(e, ld, rd, L)
        print(e, plr, init_period, 1 / wave_speed, wave_speed, collapse_time, wave_width)
        init_periods[i] = init_period
        speeds[i] = wave_speed
        collapse_times[i] = collapse_time
        wave_widths[i] = wave_width
        i += 1
    #with open('de_plrs_5_10_50_047.pkl', 'wb') as f:
    #    pickle.dump(plrs_all, f)
    # np.savetxt(f"wave_params_bec_bp_{ld}_{rd}_semi_v2.txt", np.c_[es, init_periods, speeds, collapse_times, wave_widths], comments='', header='E INIT_PERIOD SPEED COLLAPSE_TIME WAVE_WIDTH')

def show_de_sw():
    ld = 5
    rd = 10
    L = 50
    W = 20
    I = 20
    #es = np.linspace(0.455, 0.49, 7 * 8 + 1)
    es = np.array([0.47])
    for e in es:
        e_evol = de_sw_sc_ldpc(e, ld, rd, L, W, I)
        for e_evol_iter, startVN, endVN, startCN, endCN in e_evol:
            plt.plot(e_evol_iter)
            plt.axvline(startVN, color='red')
            plt.axvline(endVN, color='red')
            plt.show()
    #with open('de_plrs_5_10_50_047.pkl', 'wb') as f:
    #    pickle.dump(plrs_all, f)
    # np.savetxt(f"wave_params_bec_bp_{ld}_{rd}_semi_v2.txt", np.c_[es, init_periods, speeds, collapse_times, wave_widths], comments='', header='E INIT_PERIOD SPEED COLLAPSE_TIME WAVE_WIDTH')


def main():
    #test()
    #test_de()
    #de_irsa()
    #de_csa()
    #test_de_sc_ldpc()
    #estimate_de_speed()
    show_de_sw()

if __name__ == '__main__':
    main()
