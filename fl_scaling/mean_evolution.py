# Mean evolution (ME) equations for SC-LDPC and FA-CSA.
# ME for SC-LDPC is described in Olmos, Urbanke, 2015:
# "A Scaling Law to Predict the Finite-Length Performance of Spatially-Coupled LDPC Codes"
# IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 61, NO. 6, JUNE 2015
# APPENDIX A. EXPECTED EVOLUTION IN ONE ITERATION OF THE PD
# (PD stands for Peeling Decoding)
#
# Implemented by Roman Sokolovskii

import numpy as np
from scipy.stats import binom
from scipy.special import comb
import matplotlib.pyplot as plt
from numpy.core.umath_tests import inner1d
from tqdm import trange
import pickle
from sys import exit
from math import isclose

from fl_scaling.density_evolution import prev_window

def gen_r(g, s, m, frame_len, total_size, is_terminated, is_bounded):
    # max_degree = frame_len
    # max_degree = int(frame_len * m * s * 0.1)
    max_degree = 20
    js = np.arange(1, max_degree + 1)
    r = np.zeros((total_size, max_degree + 2))
    for u in range(total_size):
        lu = prev_window(u, frame_len) if is_bounded else frame_len
        p_transmits = g * s / frame_len / m
        num_users = m * lu
        r[u][1:max_degree + 1] = js * m * binom.pmf(js, num_users, p_transmits)
    if is_terminated:
        r[-1:-(frame_len + 1):-1, :] = r[0:frame_len, :]
    # print(np.sum(r, axis=1))
    return r


def gen_r_sc_ldpc(e, l, r, m, frame_len, total_size, is_terminated, edge_perspective=True):
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
            ra[u, j] = m * l / r * agg
            if edge_perspective:
                ra[u, j] *= j
    if is_terminated:
        ra[-1:-(frame_len + 1):-1, :] = ra[0:frame_len, :]
    return ra


# Given the position, returns the position at the termination end where the
# CN distribution is the same. It needs not only the position but also the neighborhood.
def doped_position_equivalent(l, r, frame_len, total_size, is_terminated, doping_points, pos):
    # 0. NB! This code assumes that you don't have any doping in the first dv positions!
    for i in range(frame_len + 1):
        if i in doping_points:
            print("ACHTUNG! DOPING CANNOT HAPPEN IN THE FIRST DV POSITIONS!")
            exit(-1)
    # 1. Figure out how many positions are doped in the neighborhood of pos
    # 2. Figure out how many edges are entering pos
    num_edges = 0
    lastvn = total_size - frame_len if is_terminated else total_size - 1
    lu = prev_window(pos, frame_len)
    for i in range(lu):
        vnpos = pos - i
        if vnpos <= lastvn and vnpos not in doping_points:
            num_edges += 1
    # 3. Find out which position at the termination end has the same number of edges entering it
    equiv_pos = num_edges - 1
    # 4. Return the index of that position
    return equiv_pos


def gen_r_sc_ldpc_doping(e, l, r, m, frame_len, total_size, is_terminated, doping_points=[]):
    max_degree = r
    ra = np.zeros((total_size, max_degree + 2))
    for u in range(frame_len + 1):
        lu = prev_window(u, frame_len)
        for j in range(1, max_degree + 1):
            agg = 0.0
            for d in range(j, max_degree + 1):
                if lu < frame_len:
                    pdu = binom.pmf(d, r, lu / frame_len)
                else:
                    pdu = 1 if d == r else 0
                agg += pdu * binom.pmf(j, d, e)
            ra[u, j] = j * m * l / r * agg
    if is_terminated:
        ra[-1:-(frame_len + 1):-1, :] = ra[0:frame_len, :]
    # assign probabilities with possible effects of doping in mind
    for u in range(total_size):
        equiv_pos = doped_position_equivalent(l, r, frame_len, total_size, is_terminated, doping_points, u)
        ra[u, :] = ra[equiv_pos, :]
    return ra


def avg_pr_not_fixed(frame_len, u, doping_spec):
    sum_pr_not_fixed = 0.0
    lu = prev_window(u, frame_len)
    for vnpos in range(u - lu + 1, u + 1):
        fixed_pos = doping_spec[vnpos] if vnpos in doping_spec else 0.0
        sum_pr_not_fixed += 1 - fixed_pos
    avg_pr = sum_pr_not_fixed / lu
    #print(frame_len, u, doping_spec, avg_pr)
    return avg_pr


# NB! The doping points near the termination ends are always assumed to be symmetric
# because of the mirroring in the end. Hope it's fine anyways.
def gen_r_sc_ldpc_doping_soft(e, l, r, m, frame_len, total_size, is_terminated, doping_spec=dict(), edge_perspective=True):
    max_degree = r
    ra = np.zeros((total_size, max_degree + 2))
    for u in range(total_size):
        lu = prev_window(u, frame_len)
        for j in range(1, max_degree + 1):
            agg = 0.0
            for d in range(j, max_degree + 1):
                if lu < frame_len:
                    pdu = binom.pmf(d, r, lu / frame_len)
                else:
                    pdu = 1 if d == r else 0
                pr_erased = e * avg_pr_not_fixed(frame_len, u, doping_spec)
                #print(avg_pr_not_fixed(frame_len, u, doping_spec))
                agg += pdu * binom.pmf(j, d, pr_erased)
            ra[u, j] = m * l / r * agg
            if edge_perspective:
                ra[u, j] *= j
    if is_terminated:
        ra[-1:-(frame_len + 1):-1, :] = ra[0:frame_len, :]
    return ra


def gen_v(g, m, frame_len, total_size, is_terminated=False, is_bounded=True):
    termination_adjust = 0 if not is_terminated else frame_len - 1
    boundary_adjust = 0 if is_bounded else frame_len - 1
    size = total_size - termination_adjust + boundary_adjust
    arr = g * m * np.ones(size)
    return np.pad(arr, (frame_len - 1 - boundary_adjust, frame_len - 1 + termination_adjust), mode='constant')


def gen_v_doping(g, m, frame_len, total_size, is_terminated=False, is_bounded=True, doping_points=[]):
    termination_adjust = 0 if not is_terminated else frame_len - 1
    boundary_adjust = 0 if is_bounded else frame_len - 1
    size = total_size - termination_adjust + boundary_adjust
    arr = g * m * np.ones(size)
    arr[doping_points] = 0
    return np.pad(arr, (frame_len - 1 - boundary_adjust, frame_len - 1 + termination_adjust), mode='constant')


def gen_v_doping_soft(g, m, frame_len, total_size, is_terminated=False, is_bounded=True, doping_spec=dict()):
    termination_adjust = 0 if not is_terminated else frame_len - 1
    boundary_adjust = 0 if is_bounded else frame_len - 1
    size = total_size - termination_adjust + boundary_adjust
    arr = g * m * np.ones(size)
    for pos in doping_spec:
        arr[pos] *= 1.0 - doping_spec[pos]
    return np.pad(arr, (frame_len - 1 - boundary_adjust, frame_len - 1 + termination_adjust), mode='constant')


def norm(a):
    s = np.sum(a)
    if s == 0:
        return a
    return a / s


def pr_deg_1_cn_chosen(r):
    pus = norm(r[:, 1])
    return pus


# Pr { a randomly chosen CN at a given position is of degree one }
def pr_deg_1_at_pos(r, rd):
    sum_cns = np.sum(r * np.arange(rd + 2), axis=1)
    # sum_cns = np.sum(r, axis=1)
    sum_cns[sum_cns == 0] = 1
    pdir = r[:, 1] / sum_cns
    return pdir


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def diagonals(x):
    return np.lib.stride_tricks.as_strided(x, ((x.shape[0] - x.shape[1] + 1) * x.shape[1], x.shape[1]),
                                           (x.strides[1], x.strides[0] - x.strides[1]))[x.shape[1] - 1:: x.shape[1]]


# Pr { a VN is removed in a PPD iteration }
def pr_vn_resolved_ppd(pdir, ld, is_terminated):
    if is_terminated:
        pdirw = rolling_window(pdir, ld)
    else:
        pdirw = rolling_window(np.pad(pdir, (0, ld - 1), mode='constant'), ld)
    p_vn_res = 1 - np.prod(1 - pdirw, axis=1)
    return p_vn_res


# Pr { an edge from VN pos u to CN pos m is _indirectly_ removed }
# (i.e., it's removed but it wasn't attached to a degree-one CN)
def pr_edge_indir_ppd(pdir, ld, is_terminated):
    if not is_terminated:
        pdir = np.pad(pdir, (0, ld - 1), mode="constant")
    pindir = 1 - pdir

    pindirw = rolling_window(pindir, ld)
    pindirw_prod = np.prod(pindirw, axis = 1)
    pindirw_prod_back_w = rolling_window(np.pad(pindirw_prod, (ld - 1, ld - 1), mode='constant'), ld)

    pindir_zero_guarded = np.copy(pindir)
    pindir_zero_guarded[pindir == 0] = 1

    pindirw_prod_back_w = pindirw_prod_back_w / pindir_zero_guarded[:,None]
    pindirw_prod_back_w *= -1
    # NB: The values at the edges are garbage (they are 1 instead of 0),
    # But they are multiplied by lambda = 0 at the later stage, so it's fine.
    pindirw_prod_back_w += 1
    pindirw_prod_back_w *= pindir[:,None] # Note it's not zero-guarded!
    return pindirw_prod_back_w


def z_arr(r, s, frame_len):
    rsum = np.sum(r, axis=1)
    rsumw = rolling_window(np.pad(rsum, (frame_len - 1, frame_len - 1), mode='constant'), frame_len)
    rsumw_sum = np.sum(rsumw, axis=1, keepdims=True)
    rsumw_sum[rsumw_sum == 0] = 1
    res = rsumw / rsumw_sum
    return res
    # z = np.zeros(r.shape[0] + 2 * (frame_len - 1))
    # z[frame_len - 1:-2 * (frame_len - 1)] = np.count_nonzero(rsumw, axis=1)
    # z[frame_len - 1:-2 * (frame_len - 1)] =
    # z[-2 * (frame_len - 1):] = s / frame_len
    # return z


def lam_arr(v, frame_len):
    vw = rolling_window(v, frame_len)
    vw_div = vw
    vw_sum = vw_div.sum(axis=1, keepdims=1)
    vw_sum[vw_sum == 0] = 1
    return vw_div / vw_sum


def xi_arr(lam, s, frame_len, xi):
    # mult = (s - 1) / (frame_len - 1)
    mult = s / frame_len
    # mult = 1
    # mult = (s - 1) / frame_len

    xi.fill(0)
    xi[frame_len - 1:, 0: -(frame_len - 1)] = lam
    xi.cumsum(axis=1, out=xi)
    xi[:, frame_len:] = xi[:, frame_len:] - xi[:, :-frame_len]
    xi *= mult
    xi[frame_len - 1: -(frame_len - 1), frame_len - 1] = 1
    return xi


def update_r(r, pu, xi, frame_len):
    s = 3  # TODO: add as a parameter
    # mult = s / frame_len
    mult = 1

    puw = rolling_window(np.pad(pu, (frame_len - 1, frame_len - 1), mode='constant'), 2 * frame_len - 1)
    xid = diagonals(xi)
    ks = inner1d(puw, xid)
    ks -= pu

    diffs = np.multiply(np.diff(r), np.arange(r.shape[1] - 1))
    sums = np.sum(r, axis=1)[:, None]
    sums[sums == 0] = 1
    diffs /= sums
    diffs *= mult
    np.multiply(diffs, ks[:, None], out=diffs)

    rsum_before = np.sum(r)
    r[:, :-1] += diffs
    r[:, 1] -= mult * pu
    # rsum_after = np.sum(r)
    # if abs(abs(rsum_before - rsum_after) - 3) > 1e-8:
    #     print("OMG CN crap!", rsum_before - rsum_after)
    np.clip(r, 0, None, out=r)
    return r


def update_v(v, pu, lam, frame_len):
    lamd = diagonals(lam)
    puw = rolling_window(np.pad(pu, (0, frame_len - 1), mode='constant'), frame_len)
    ks = inner1d(puw, lamd)
    # s = 3
    # mult = s / frame_len
    # mult = (s - 1) / (frame_len - 1)
    mult = 1
    ks *= mult
    v[frame_len - 1: -(frame_len - 1)] -= ks
    # if abs((np.sum(ks) - 1)) > 1e-8:
    #     print("OMG VN crap!", np.sum(ks) - 1)
    # v[v < 1e-5] = 0
    np.clip(v, 0, None, out=v)
    return v


def average_deg_cns(r, deg):
    return sum(r[:, deg])


def pad(a, before, after=None):
    pattern = (before, after) if after else (before, 0)
    return np.pad(a, pattern, mode='constant')


def steps_internal(r, v, s, frame_len, total_size, t, numsteps):
    xi = np.zeros((total_size + 2 * (frame_len - 1), 2 * frame_len - 1))

    deg1s = np.zeros(numsteps)
    # variances = np.zeros(numsteps)
    # r_dist = np.zeros((numsteps, r.shape[0]))
    # r_total = np.zeros((numsteps, r.shape[0], r.shape[1]))
    v_total = np.zeros((numsteps, v.shape[0]))
    # delta_prev = 0

    # sstime = None
    # middle = int(r.shape[0] / 2)
    # threshold = 1e-9

    for l in t:
    # for l in range(numsteps):

        # if l % 100 == 0:
        #     plt.plot(np.arange(total_size), v[frame_len-1:])
        #     plt.show()

        # r_sum = np.sum(r, axis=1)
        # v_window = rolling_window(v, frame_len)
        # v_window_scaled = v_window * s / frame_len
        # v_window_scaled_sum = np.sum(v_window_scaled, axis=1)
        # delta = r_sum - v_window_scaled_sum[:-(frame_len - 1)]
        # delta_diff = delta_prev - delta
        # delta_prev = delta

        # # if l % 200 == 0:
        # #     print(np.sum(delta))
        # # if True or l == 0 or not np.allclose(r_sum, v_window_scaled_sum[:-(frame_len - 1)], atol=0.01, equal_nan=True):
        # if not np.allclose(r_sum, v_window_scaled_sum[:-(frame_len - 1)], atol=0.01, equal_nan=True):
        #     if l % 1000 == 0:
        #         plt.plot(np.arange(delta.shape[0]), delta)
        #         plt.title(l)
        #         plt.show()

        pu = pr_deg_1_cn_chosen(r)
        lam = lam_arr(v, frame_len)
        xi_arr(lam, s, frame_len, xi)

        update_r(r, pu, xi, frame_len)
        update_v(v, pu, lam, frame_len)

        # if not sstime:
        #     if r[middle,1] < threshold:
        #         sstime = l
        #         print(sstime)
        #         #break

        r1s = average_deg_cns(r, 1)
        deg1s[l] = r1s
        # if (l % 1000 == 0):
        #     print(str(l) + ": " + str(r1s))
        #variances[l] = variance
        # r_dist[l, :] = r[:, 1]
        # r_total[l, :, :] = r[:, :]
        v_total[l, :] = v
        # t.set_description("R1: %.2f" % r1s)
        # print(np.sum(r, axis=1))
    # return deg1s, r_dist, r_total, v_total
    # return deg1s, None, None, None, variances
    # return deg1s, None, None, None, None
    # return deg1s, r_dist, None, None, None
    # return deg1s, r_dist, None, v_total, None
    return deg1s, None, None, v_total, None
    # return deg1s, None, r_total, v_total, variances


def steps_internal_parallel_pd(r, v, ld, rd, N, L, total_size, is_terminated, t, numsteps):
    deg1s = np.zeros(numsteps + 1)
    # r_dist = np.zeros((numsteps + 1, r.shape[0]))
    # r_total = np.zeros((numsteps + 1, r.shape[0], r.shape[1]))
    v_total = np.zeros((numsteps + 1, v.shape[0]))

    ns = np.arange(rd + 1)
    ks = np.arange(rd + 1)
    nmks = ns[:,None] - ks[None,:]
    nmks[nmks < 0] = 0  # This doesn't change the math but avoids some numerical unpleasantries.

    combs = np.array([[comb(n, k, exact=True) for k in ks] for n in ns])

    pr_cn_changes_deg = np.zeros((total_size, rd + 1, rd + 1))  # pos, deg_from, deg_to
    temp_arr = np.zeros((total_size, rd + 1, rd + 1))

    deg1s[0] = average_deg_cns(r, 1)
    # r_dist[0, :] = r[:, 1]
    # r_total[0, :, :] = r[:, :]
    v_total[0, :] = v

    #t.set_description("R1: %.2f" % deg1s[0])

    for l in t:
        # # A consistency check: the number of outgoing edges from VNs
        # # should match the number of incoming edges from CNs
        # r_sum = np.sum(r * np.arange(rd + 2), axis=1)
        # v_window = rolling_window(v, ld)
        # v_window_scaled = v_window
        # v_window_scaled_sum = np.sum(v_window_scaled, axis=1)
        # delta = r_sum - v_window_scaled_sum[:-(ld - 1)]

        # if not np.allclose(r_sum, v_window_scaled_sum[:-(ld - 1)], atol=0.01, equal_nan=True):
        #     if l % 1 == 0:
        #         plt.plot(np.arange(delta.shape[0]), delta)
        #         #plt.plot(delta[20:30])
        #         #print(np.mean(delta[20:30]))
        #         plt.xlim(left=0)
        #         #plt.ylim(bottom=0)
        #         plt.title(l)
        #         plt.show()

        pdir = pr_deg_1_at_pos(r, rd)

        lam = lam_arr(v, ld)

        p_vn_res = pr_vn_resolved_ppd(pdir, ld, is_terminated)

        p_edge_indir = pr_edge_indir_ppd(pdir, ld, is_terminated)
        if not is_terminated:
            p_edge_indir = p_edge_indir[:-(ld-1),:]
        p_edge_cnpos_indir = inner1d(p_edge_indir, lam[:-(ld-1),:])
        notpdir = 1 - pdir
        notpdir[notpdir == 0] = 1
        p_edge_cnpos_indir = p_edge_cnpos_indir / notpdir

        # pr_cn_changes_deg = combs[None,:] * np.power.outer(1 - ps, ks[None,:]) * np.power.outer(ps, nmks)
        # Same thing but in constant memory
        np.power.outer(1 - p_edge_cnpos_indir, ks[None,:], out=temp_arr)
        np.power.outer(p_edge_cnpos_indir, nmks, out=pr_cn_changes_deg)
        np.multiply(pr_cn_changes_deg, temp_arr, out=pr_cn_changes_deg)
        np.multiply(pr_cn_changes_deg, combs[None,:], out=pr_cn_changes_deg)

        np.multiply(pr_cn_changes_deg, r[:,:-1,None], out=pr_cn_changes_deg)
        rdeltas = np.sum(pr_cn_changes_deg, axis=1) - np.sum(pr_cn_changes_deg, axis=2)
        rdeltas[:,1] = np.sum(pr_cn_changes_deg[:,2:,1], axis=1) - r[:,1]

        r[:,1:-1] += rdeltas[:,1:]

        # This should happen _after_ the lam array is constructed
        v[ld - 1 : ld - 1 + L] -= v[ld - 1 : ld - 1 + L] * p_vn_res

        r1s = average_deg_cns(r, 1)
        deg1s[l + 1] = r1s

        # r_dist[l + 1, :] = r[:, 1]
        # r_total[l + 1, :, :] = r[:, :]
        v_total[l + 1, :] = v
        #t.set_description("R1: %.2f" % r1s)
        # print(np.sum(r, axis=1))

    # return deg1s, r_dist, r_total, v_total
    # return deg1s, None, None, None
    # return deg1s, None, None, None
    # return deg1s, r_dist, None, None
    # return deg1s, r_dist, None, v_total
    return deg1s, None, None, v_total
    # return deg1s, None, r_total, v_total


def steps_fa_csa(g, s, m, w, frame_len, is_terminated=False, is_bounded=True):
    total_size = w + frame_len - 1 if is_terminated else w
    # total_size = w * frame_len
    numsteps = int(total_size * g * m)
    # t = trange(numsteps, ncols=120)
    t = range(numsteps)

    r = gen_r(g, s, m, frame_len, total_size, is_terminated, is_bounded)
    # r = gen_r_sc_csa(g, s, m, frame_len, total_size, is_terminated)
    v = gen_v(g, m, frame_len, total_size, is_terminated, is_bounded)

    deg1s, r_dist, r_total, v_total, variances = steps_internal(r, v, s, frame_len, total_size, t, numsteps)

    # with open('r1_sc_csa_1000_086_100_nonterminated_theory.pkl', 'wb') as f:
    #     pickle.dump([deg1s, r_dist, r_total, v_total], f)
    # #     pickle.dump([deg1s, r_dist], f)
    # plt.semilogy(np.arange(deg1s.shape[0]), deg1s)
    # # plt.xlim(xmin=0, xmax=total_size * g)
    # plt.ylim(ymin = 1e1)
    # plt.minorticks_on()
    # plt.grid(which='both')
    # plt.show()
    return deg1s, r_dist, r_total, v_total


# doping_points is for hard doping
# doping spec : { doped_position : fraction_fixed in [0,1] } is for soft doping
def steps_sc_ldpc(e, s, m, w, frame_len, is_terminated, rd=None, is_bounded=True, doping_points=[]):
    num_dopings = len(doping_points)
    is_hard_doping = isinstance(doping_points, list)

    total_size = (w + frame_len - 1) if is_terminated else w
    numsteps = int(total_size * m * e)
    # t = trange(numsteps, ncols=120)
    t = range(numsteps)

    ld = s
    if not rd:
        rd = 2 * ld

    if num_dopings == 0:
        r = gen_r_sc_ldpc(e, ld, rd, m, frame_len, total_size, is_terminated)
        v = gen_v(e, m, frame_len, total_size, is_terminated, is_bounded)
    else:
        if is_hard_doping:
            r = gen_r_sc_ldpc_doping(e, ld, rd, m, frame_len, total_size, is_terminated, doping_points)
            v = gen_v_doping(e, m, frame_len, total_size, is_terminated, is_bounded, doping_points)
        else:  # soft doping
            r = gen_r_sc_ldpc_doping_soft(e, ld, rd, m, frame_len, total_size, is_terminated, doping_points)
            v = gen_v_doping_soft(e, m, frame_len, total_size, is_terminated, is_bounded, doping_points)

    deg1s, r_dist, r_total, v_total, variances = steps_internal(r, v, s, frame_len, total_size, t, numsteps)

    # plt.semilogy(np.arange(numsteps), deg1s)
    # # # plt.ylim(ymax=10, ymin=1e-2)
    # plt.xlim(left=0, right=total_size * m * 0.5)
    # plt.minorticks_on()
    # plt.grid(which='both')
    # plt.show()
    # # # plt.plot(np.arange(variances.shape[0]), variances)
    # # # plt.show()

    # with open('r1_sc_ldpc_5_10_100_1000_49_51_53_0497_terminated_theory_full.pkl', 'wb') as f:
        # pickle.dump([deg1s, r_dist, r_total, v_total, variances], f)
    # term_suffix = "terminated" if is_terminated else "non_terminated"
    # with open(f'r1_sc_ldpc_{ld}_{rd}_{w}_{m}_{e}_{term_suffix}_pd_me.pkl', 'wb') as f:
    #     pickle.dump([deg1s, r_dist, r_total, v_total], f)
    return deg1s, r_dist, r_total, v_total, variances


def steps_sc_ldpc_parallel_pd(e, N, L, ld, rd=None, is_terminated=True, is_bounded=True):
    total_size = (L + ld - 1) if is_terminated else L
    numsteps = 1000  # TODO: How do I calculate this value?
    # t = trange(numsteps, ncols=120)
    t = range(numsteps)

    if not rd:
        rd = 2 * ld

    # The initial conditions for mean evolution are the same for regular and parallel PD,
    # except for the fact that in parallel PD we track the number of CNs of degree j -- not
    # edges adjacent to them -- hence edge_perspective=False.
    r = gen_r_sc_ldpc(e, ld, rd, N, ld, total_size, is_terminated, edge_perspective=False)
    v = gen_v(e, N, ld, total_size, is_terminated, is_bounded)

    deg1s, r_dist, r_total, v_total = steps_internal_parallel_pd(r, v, ld, rd, N, L, total_size, is_terminated, t, numsteps)

    # plt.semilogy(np.arange(numsteps + 1), deg1s)
    # plt.ylim(top=1e3, bottom=1e1)
    # plt.xlim(left=0)
    # plt.minorticks_on()
    # plt.grid(which='both')
    # plt.show()

    # term_suffix = "terminated" if is_terminated else "non_terminated"
    # with open(f'r1_sc_ldpc_{ld}_{rd}_{L}_{N}_{e}_{term_suffix}_ppd_me_full.pkl', 'wb') as f:
    # # with open(f'r1_sc_ldpc_{ld}_{rd}_{L}_{N}_{e}_{term_suffix}_ppd_me.pkl', 'wb') as f:
    #     pickle.dump([deg1s, r_dist, r_total, v_total], f)
    return deg1s, r_dist, r_total, v_total


def main():
    # steps_fa_csa(g=0.8, s=3, m=1, w=40, frame_len=200)
    # steps_fa_csa(g=0.84, s=3, m=100, w=200, frame_len=20, is_terminated=False)
    # steps_fa_csa(g=0.84, s=3, m=200, w=100, frame_len=10, is_terminated=True, is_bounded=True)
    # steps_fa_csa(g=0.86, s=3, m=1000, w=100, frame_len=3, is_terminated=False, is_bounded=True)
    # steps_fa_csa(g=0.88, s=3, m=500, w=200, frame_len=3, is_terminated=False, is_bounded=True)
    # steps_fa_csa(g=0.84, s=3, m=1, w=40 * 200, frame_len=1000)
    # steps_fa_csa(g=0.8, s=2, m=1, w=3, frame_len=3)
    # for i in [1,2,3,4,5,6,7,8,9,10]:
    #     steps_sc_ldpc(e=0.46, s=3, m=i * 10, w=50 * 10, frame_len=10, is_terminated=True)
    # steps_sc_ldpc(e=0.4881 - 0.01, s=3, m=100000, w=100, frame_len=3, is_terminated=True)
    # steps_sc_ldpc(e=0.4781, s=3, m=100, w=100, frame_len=3, is_terminated=False)
    # steps_sc_ldpc(e=0.4850 - 1e-2 , s=4, m=100, w=100, frame_len=4, is_terminated=True)
    # steps_sc_ldpc(e=0.3175, s=4, m=9999, w=50, frame_len=4, is_terminated=False,rd=12)
    # steps_sc_ldpc(e=0.305, s=3, m=9999, w=50, frame_len=3, is_terminated=False, rd=9)
    # steps_sc_ldpc(e=0.4875, s=5, m=2000, w=50, frame_len=5, is_terminated=False)
    # steps_sc_ldpc(e=0.485, s=3, m=100, w=50, frame_len=3, is_terminated=True)
    # steps_sc_ldpc(e=0.46, s=4, m=50, w=100, frame_len=10, is_terminated=False)
    # steps_sc_ldpc(e=0.46, s=3, m=100, w=50, frame_len=4, is_terminated=False)

    # steps_sc_ldpc(e=0.475, s=5, m=10000, w=50, frame_len=5, is_terminated=True)
    # steps_sc_ldpc(e=0.485, s=6, m=10000, w=50, frame_len=6, is_terminated=False)
    # steps_sc_ldpc(e=0.475, s=5, m=1000, w=101, frame_len=5, is_terminated=True, doping_points=[50])
    # steps_sc_ldpc(e=0.475, s=5, m=1000, w=102, frame_len=5, is_terminated=True, doping_points=[50,51])
    # steps_sc_ldpc(e=0.475, s=5, m=1000, w=103, frame_len=5, is_terminated=True, doping_points=[50,51,52])
    # steps_sc_ldpc(e=0.475, s=5, m=1000, w=104, frame_len=5, is_terminated=True, doping_points=[50,51,52,53])
    # steps_sc_ldpc(e=0.478, s=5, m=1000, w=104, frame_len=5, is_terminated=True, doping_points=[50,51,52,53])

    # steps_sc_ldpc(e=0.44, s=5, m=1000, w=102, frame_len=5, is_terminated=True)
    # steps_sc_ldpc(e=0.46, s=5, m=1000, w=102, frame_len=5, is_terminated=True, doping_points=[50,51])

    # steps_sc_ldpc(e=0.475, s=5, m=1000, w=103, frame_len=5, is_terminated=True)
    # steps_sc_ldpc(e=0.475, s=5, m=1000, w=103, frame_len=5, is_terminated=True, doping_points=[50,51,52])
    # steps_sc_ldpc(e=0.475, s=5, m=1000, w=103, frame_len=5, is_terminated=True, doping_points={50 : 1.0, 51 : 1.0, 52 : 1.0})
    # steps_sc_ldpc(e=0.475, s=5, m=1000, w=103, frame_len=5, is_terminated=True, doping_points={49 : 0.8, 50 : 0.2, 51 : 0.8, 52 : 0.2, 53 : 0.8})
    # steps_sc_ldpc(e=0.497, s=5, m=1000, w=103, frame_len=5, is_terminated=True, doping_points=[49, 51, 53])
    # steps_sc_ldpc(e=0.48, s=5, m=1000, w=103, frame_len=5, is_terminated=True)
    # steps_sc_ldpc(e=0.478, s=5, m=1000, w=103, frame_len=5, is_terminated=True, doping_points=[50,51,52])
    # steps_sc_ldpc(e=0.48, s=5, m=1000, w=100, frame_len=5, is_terminated=True)
    # steps_sc_ldpc(e=0.478, s=5, m=1000, w=100, frame_len=5, is_terminated=True)

    # steps_sc_ldpc(e=0.47, s=3, m=1000, w=101, frame_len=3, is_terminated=True, doping_points=[50])
    # steps_sc_ldpc(e=0.475, s=3, m=1000, w=101, frame_len=3, is_terminated=True, doping_points=[50])

    # steps_sc_ldpc_parallel_pd(e=0.46, N=2500, L=50, ld=5, is_terminated=False)
    steps_sc_ldpc_parallel_pd(e=0.485, N=5000, L=50, ld=5, is_terminated=False)
    # steps_sc_ldpc_parallel_pd(e=0.4675, N=1000, L=50, ld=5, is_terminated=False)
    # steps_sc_ldpc_parallel_pd(e=0.475, N=5000, L=50, ld=5, is_terminated=False)
    # steps_sc_ldpc(e=0.4675, s=5, m=1000, w=50, frame_len=5, is_terminated=False)
    # steps_sc_ldpc(e=0.465, s=5, m=5000, w=50, frame_len=5, is_terminated=False)
    # steps_sc_ldpc(e=0.46, s=5, m=1000, w=50, frame_len=5, is_terminated=False)
    # steps_sc_ldpc_parallel_pd(e=0.5, N=1000, L=2, ld=2, is_terminated=False)


def test_doped_positions():
    if doped_position_equivalent(5,10,5,101,True,[50],50) != 3:
        print("WRONG!")
        return
    poss = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    ress = [0,1,2,3,3,3,2,2,2,2,3,2,1,1,1,2,3,3,2,1,0]
    for pos in poss:
        x = doped_position_equivalent(4,8,4,21,True,[6,11,12],pos)
        if x != ress[pos]:
            print(f"POS: {pos} IS WRONG! (not {ress[pos]} but {x} instead)")
            return
    print("OK!")


def test_avg_pr_not_fixed():
    frame_len = 3
    doping_spec = { 0 : 0.1, 1 : 0.2, 6 : 0.3, 7 : 0.4, 8 : 0.2, 9 : 0.4, 10 : 0.3 }
    ress = [0.9, 0.85, 0.9, 1 - 0.2 / 3, 1, 1, 0.9, 1 - 0.7 / 3, 0.7, 2 / 3, 0.7]
    for u in range(10):
        x = avg_pr_not_fixed(frame_len, u, doping_spec)
        res = ress[u]
        if not isclose(x, res):
            print(f"FAIL: avg_pr_not_fixed({frame_len},{u},{doping_spec}) = {x} != {res}")
            return
    print("PASS")


def test_soft_hard_doping_equivalence():
    frame_len = 3
    rsoft = gen_r_sc_ldpc_doping_soft(0.45, 3, 6, 1000, 3, 20, True, doping_spec={10 : 1.0})
    rhard = gen_r_sc_ldpc_doping(0.45, 3, 6, 1000, 3, 20, True, doping_points=[10])
    vhard = gen_v_doping(0.45, 1000, 3, 20, True, True, [10])
    vsoft = gen_v_doping_soft(0.45, 1000, 3, 20, True, True, {10 : 1.0})
    if not np.allclose(rsoft, rhard):
        print("FAIL! They are not the same!")
        print(rsoft - rhard)
        return
    if not np.allclose(vsoft, vhard):
        print("FAIL! They are not the same!")
        print(vsoft - vhard)
        return
    rsoft = gen_r_sc_ldpc_doping_soft(0.45, 5, 10, 1000, 5, 30, True, doping_spec={10 : 1.0, 12 : 1.0, 14 : 1.0})
    rhard = gen_r_sc_ldpc_doping(0.45, 5, 10, 1000, 5, 30, True, doping_points=[10, 12, 14])
    vhard = gen_v_doping(0.45, 1000, 5, 30, True, True, [10, 12, 14])
    vsoft = gen_v_doping_soft(0.45, 1000, 5, 30, True, True, {10 : 1.0, 12 : 1.0, 14 : 1.0})
    if not np.allclose(rsoft, rhard):
        print("FAIL! They are not the same!")
        print(rsoft - rhard)
        return
    if not np.allclose(vsoft, vhard):
        print("FAIL! They are not the same!")
        print(vsoft - vhard)
        return
    print("PASS")


def test_parallel_pd():
    ld = 3
    a = 1.0 - (np.arange(5) + 1.0)
    nvns = gen_v(0.5, 10, 3, 5, True, True)
    lam = lam_arr(nvns, ld)
    print(lam[:-(ld-1),:])
    pvn = pr_vn_resolved_ppd(a, ld, True)
    p_edge_indir = pr_edge_indir_ppd(a, ld, True)
    print(p_edge_indir)
    print(inner1d(p_edge_indir, lam[:-(ld-1),:]))
    #r = gen_r_sc_ldpc(e, ld, rd, N, ld, total_size, is_terminated, edge_perspective=False)
    r = gen_r_sc_ldpc(0.5, 3, 6, 10, 3, 5, True, edge_perspective=False)
    print(r)


if __name__ == '__main__':
    main()
    # test_doped_positions()
    # test_avg_pr_not_fixed()
    # test_soft_hard_doping_equivalence()
    # test_parallel_pd()
