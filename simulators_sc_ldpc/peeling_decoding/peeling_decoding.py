# This is an implementation of peeling decoding for LDPC, semi-structured SC-LDPC,
# protograph-based SC-LDPC, and SC-CSA (Coded Slotted Aloha).
#
# VN doping (both soft and hard) is implemented as described in
# R. Sokolovskii, A. Graell i Amat, and F. Br\"annstr\"om,
# "On doped SC- LDPC codes for streaming"
# IEEE Commun. Lett., vol. 25, no. 7, pp. 2123–2127, Jul. 2021.
#
# The code was originally written with CSA in mind, so the terminology used is more
# appropriate for multiple access than for LDPC codes (over the BEC).
# However, there is a one-to-one mapping between the terms used in CSA and those
# used in coding:
# User <-> Bit or Variable Node (VN)
# Slot <-> Constraint or Check Node (CN)
# Load g <-> Erasure probability e (epsilon)
# Successive Interference Cancellation (SIC) round <-> Peeling Decoding Iteration
#
# Note: M is used here instead of N to mean the number of VNs per spatial position
#
# Implemented by Roman Sokolovskii

import numpy as np
from scipy import stats
import scipy.io
from collections import namedtuple
import sys
import random
from tqdm import trange
import pickle

import ldpc
import sc_ldpc
import sc_ldpc_protograph
from covariance import init_cov, update_cov, get_cov

# Adding the grandparent directory to PATH to import fl_scaling.est_scaling_params
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandparentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, grandparentdir) 

from fl_scaling.est_scaling_params import *


np.set_printoptions(precision=4, suppress=True, linewidth=200, sign=' ')


class User(namedtuple("user_proto", ["uid", "birthday", "transmissions", "recovered", "k"])):
    def __hash__(self): return hash(self.uid)


def gen_slots(vn, frame_len, is_first_fixed):
    num_bursts = vn.rvs()
    random_indices = gen_indices(frame_len, num_bursts, is_first_fixed)
    return random_indices


def gen_slots_sc_csa(l, w, frame_len):
    frame_indices = gen_indices(w, l, is_first_fixed=False)
    return frame_indices * frame_len + np.random.randint(frame_len, size=l)


# copied from Alex's simulation for debugging purposes
vc_un = 1


def unif():
    global vc_un
    mm = 0x7FFFFFFF
    a = 16807
    q = 127773
    r = 2836

    s = vc_un // q
    vc_un = a * (vc_un - q * s) - r * s
    if vc_un <= 0:
        vc_un += mm
    return vc_un / mm


def randint(range_len):
    # ind = np.random.randint(range_len)
    r = unif() * range_len
    ind = int(r)
    return ind


# designed to the case where range_len is much greater than number of indices needed
def gen_indices(range_len, num_indices, is_first_fixed):
    assert num_indices <= range_len
    if num_indices == range_len:
        return np.arange(range_len)
    res = np.zeros(num_indices, dtype='int')
    used = set()
    if is_first_fixed:
        res[0] = 0
        used.add(0)
    for i in np.arange(is_first_fixed, num_indices):
        while True:
            ind = randint(range_len)
            if ind not in used:
                break
        res[i] = ind
        used.add(ind)
    res.sort()
    return res


def gen_user(uid, t, k, vn, frame_len, is_first_fixed, is_sync):
    frame_len_slices = k * frame_len
    start = t if not is_sync else frame_sync_offset(t, frame_len_slices)
    slots = start + gen_slots(vn, frame_len_slices, is_first_fixed)
    rec = set()
    return User(uid, t, slots, rec, k)


# Throughout this simulation, frame_len means the length of the local frame for a user.
# A user selects one slot out of frame_len slots and sends a copy of its message.
# That is, the overall span of the user is frame_len * w
# l - number of copies to transmit
# w - number of frames to select from
def gen_user_sc_csa(uid, t, l, w, frame_len):
    assert w >= l
    start = frame_sync_offset(t, frame_len)
    slots = start + gen_slots_sc_csa(l, w, frame_len)
    return User(uid, t, slots, set(), 1)


def gen_users_ldpc(l, r, M, e):
    uid = 0
    cns_per_pos = int(l / r * M)
    transmissions = ldpc.gen_slots(l, r, M)
    erasure = np.random.rand(M) <= e
    erased_transmissions = transmissions[erasure]
    for transmission in erased_transmissions:
        assert len(transmission) == l
        # position corresponds to the earliest transmission
        # multiply because the birthday is used in the clean method
        pos = int(transmission[0] / cns_per_pos) * cns_per_pos
        user = User(uid, pos, transmission, set(), 1)
        uid += 1
        yield user


def gen_users_sc_ldpc(l, r, L, M, is_tail_biting, e):
    uid = 0
    cns_per_pos = int(l / r * M)
    if is_tail_biting:
        transmissions = sc_ldpc.gen_slots_tail_biting(l, r, L, M)
    else:
        transmissions = sc_ldpc.gen_slots(l, r, L, M)
    erasure = np.random.rand(L * M) <= e
    erased_transmissions = transmissions[erasure]
    for transmission in erased_transmissions:
        assert len(transmission) == l
        # position corresponds to the earliest transmission
        # multiply because the birthday is used in the clean method
        pos = int(transmission[0] / cns_per_pos) * cns_per_pos
        user = User(uid, pos, transmission, set(), 1)
        uid += 1
        yield user


def gen_users_sc_ldpc_doping(l, r, L, M, doping_points, is_tail_biting, e):
    is_soft = isinstance(doping_points, dict)
    uid = 0
    cns_per_pos = int(l / r * M)
    if is_tail_biting:
        transmissions = sc_ldpc.gen_slots_tail_biting(l, r, L, M)
    else:
        transmissions = sc_ldpc.gen_slots(l, r, L, M)
    erasure = np.random.rand(L * M) <= e
    if is_soft:
        for vn_position in doping_points:
            alpha = doping_points[vn_position]
            num_fixed = int(alpha * M)
            #print(vn_position, M, num_fixed, vn_position * M, vn_position * M + num_fixed)
            # fixed = np.arange(M)
            # fixed = np.random.permutation(fixed)[0 : num_fixed]
            # erasure[vn_position * M + fixed] = False
            erasure[vn_position * M : vn_position * M + num_fixed] = False
    erased_transmissions = transmissions[erasure]
    for transmission in erased_transmissions:
        assert len(transmission) == l
        # position corresponds to the earliest transmission
        # multiply because the birthday is used in the clean method
        pos = int(transmission[0] / cns_per_pos) * cns_per_pos
        pos_chain = int(transmission[0] / cns_per_pos)
        # In soft doping, this has already been taken care of
        if is_soft or pos_chain not in doping_points:
            user = User(uid, pos, transmission, set(), 1)
            uid += 1
            yield user


def gen_users_sc_ldpc_protograph(l, r, L, M, is_tail_biting, e):
    uid = 0
    for vn_position in range(L):
        cns_per_pos = int(l / r * M)
        seed = vn_position * cns_per_pos
        vns = seed + sc_ldpc_protograph.gen_slots_from_position(l, r, M)
        if is_tail_biting:
            vns = vns % L
        erasure = np.random.rand(M) <= e
        erased_transmissions = vns[erasure]
        for transmission in erased_transmissions:
            assert len(transmission) == l
            pos = int(transmission[0] / cns_per_pos) * cns_per_pos
            user = User(uid, pos, transmission, set(), 1)
            uid += 1
            yield user


def gen_users_sc_ldpc_protograph_doping(l, r, L, M, doping_points, is_tail_biting, e):
    uid = 0
    is_soft = isinstance(doping_points, dict)
    for vn_position in range(L):
        cns_per_pos = int(l / r * M)
        seed = vn_position * cns_per_pos
        vns = seed + sc_ldpc_protograph.gen_slots_from_position(l, r, M)
        if is_tail_biting:
            vns = vns % L
        erasure = np.random.rand(M) <= e
        if is_soft and vn_position in doping_points:
            alpha = doping_points[position]
            num_fixed = int(alpha * M)
            erasure[0 : num_fixed] = False
        erased_transmissions = vns[erasure]
        for transmission in erased_transmissions:
            assert len(transmission) == l
            pos = int(transmission[0] / cns_per_pos) * cns_per_pos
            pos_chain = int(transmission[0] / cns_per_pos)
            # In soft doping, this has already been taken care of
            if is_soft or pos_chain not in doping_points:
                user = User(uid, pos, transmission, set(), 1)
                uid += 1
                yield user


def frame_sync_offset(t, frame_len):
    return (t // frame_len) * frame_len + frame_len


def gen_poisson_arrival(lam):
    return np.random.poisson(lam)


# always return a set
def recovered(users):
    return {u for u in users if is_recovered(u)}


def is_recovered(user):
    return len(user.recovered) >= user.k


def decode_slice(user, t):
    user.recovered.add(t)
    return user


def head(xs):
    return next(iter(xs))


# Iteratively performs successive interference cancellation while
# possible. Returns the list of decoded users.
def sic_round(schedule, t):
    decoded = set()
    # if conflict or empty
    if t not in schedule or len(schedule[t]) != 1:
        return decoded
    # ds is a temporary set of decoded users for which SIC has not
    # yet been performed
    single_user = head(schedule[t])
    decode_slice(single_user, t)
    ds = recovered(schedule[t])
    decoded |= ds
    while ds:
        d = ds.pop()
        # decoded after SIC the first success user
        revealed = subtract_interference(schedule, d, t)
        ds |= revealed
        decoded |= revealed
    return decoded


# delete user from every time slot he has been part of.
# next, return the set of users that were previously involved
# in the conflict of cardinality 2 with the current user
# and were left 'revealed' - alone; hence, which can now be decoded.
def subtract_interference(schedule, user, t):
    # print("DECODED: ", t, user)
    revealed = set()
    slot_indices = user.transmissions
    for slot_idx in slot_indices:
        if slot_idx not in schedule:
            print("WARN: slot_idx not in schedule. User has transmissions before last clean? user:", user, "t:", t,
                  "slot_idx:", slot_idx, "t - slot_idx =", t - slot_idx)
            continue
        slot = schedule[slot_idx]
        slot.remove(user)
        # we can not decode transmissions in the future
        if len(slot) == 1 and slot_idx <= t:
            rev = head(slot)
            decode_slice(rev, slot_idx)
            revealed |= slot
        if len(slot) == 0:
            schedule.pop(slot_idx)
    # return only those filtered ones which are fully decoded
    return recovered(revealed)


# NB: sorted dict would facilitate iteration through low-density loads,
# but insertion would be more difficult, and everything would be organized
# on a stage-basis (we would need to generate all the users first, because
# only in this case we know whether we can skip the time-slot or not)
def build_schedule(users):
    return add_to_schedule(empty_schedule(), users)


def empty_schedule():
    return {}


def add_to_schedule(schedule, users):
    for user in users:
        for t in user.transmissions:
            if t not in schedule:
                schedule[t] = set()
            schedule[t].add(user)
    return schedule


def print_schedule(schedule):
    if not schedule:
        print("EMPTY SCHEDULE")
    else:
        for t in sorted(schedule.keys()):
            print(t, [u.uid for u in schedule[t]])


def simulate_fa_csa(k, lam, vn_deg, vn_pr, is_first_fixed, frame_len, max_time, is_sync, is_report):
    max_failures = int(1e7)
    report_ivl = int(1e5)

    clean_ivl = frame_len * k * 20

    vn_rv = stats.rv_discrete(name='vn_deg', values=(vn_deg, vn_pr))
    schedule = empty_schedule()
    uid = 0
    total_generated = 0
    total_decoded = 0

    fers = 0

    generated_old = 0

    lost_old = 0

    plr_cur = 0
    fer_cur = 0

    print("lam:", lam, "k:", k, "vn_deg:", vn_deg, "vn_pr:", vn_pr, "FF:", is_first_fixed, "SYNC:", is_sync,
          "frame_len:", frame_len, "max_time:", max_time, "max_failures:", max_failures, "report_ivl:", report_ivl,
          "clean_ivl:", clean_ivl)
    print('{:<12} {:>12} {:>12} {:>12} {:>12} {:>12}'.format("T", "SCHEDULE_SIZE", "TOTAL USERS", "DECODED", "FAILED",
                                                             "PLR"))
    for t in range(max_time):
        num_newborns = gen_poisson_arrival(lam / k)
        # num_newborns = 1 if np.random.random() <= (lam / k) else 0
        total_generated += num_newborns
        newborns = []
        for i in range(num_newborns):
            # newborn = gen_user(uid, t, k, vn_rv, frame_len, is_first_fixed, is_sync)
            newborn = gen_user_sc_csa(uid, t, 3, 3, frame_len)
            newborns.append(newborn)
            uid += 1
        add_to_schedule(schedule, newborns)
        decoded = sic_round(schedule, t)
        total_decoded += len(decoded)

        if t % report_ivl == 0 and t != 0:
            report(schedule, t, total_generated, total_decoded, lost_old, generated_old)

        if t % clean_ivl == 0 and t != 0:
            lost_prev, fers_prev = clean(schedule, t, clean_ivl, frame_len, k, is_report)
            lost_old += lost_prev
            fers += fers_prev

            plr_cur = calc_plr(lost_old, generated_old)
            fer_cur = fers / ((t - clean_ivl) // (frame_len * k) + 1)
            if lost_old >= max_failures:
                print("max_failures break:", max_failures)
                break
            # generated_delta = total_generated - generated_old
            # report(schedule, t, total_generated, total_decoded, lost_prev, generated_delta)

            generated_old = total_generated  # for next clean

    return plr_cur, fer_cur


# noinspection SpellCheckingInspection
def simulate_peeling_decoder(k, lam, vn_deg, vn_pr, is_first_fixed, frame_len, _max_time, is_sync, is_report, total_size):
    vn_rv = stats.rv_discrete(name='vn_deg', values=(vn_deg, vn_pr))

    is_terminated = False
    is_bounded = True  # non-bounded means that we start at the middle and cannot recover previous transmissions
    # users that arrive first will transmit in next frame, so there is a delay by 1
    termination_tail = 1 if not is_terminated else frame_len * 3
    boundary_head = 0 if is_bounded else frame_len * 4

    # num_repeats = int(1e4)
    # num_repeats = 100
    num_repeats = 500
    num_pd_steps = total_size + 1
    num_positions = boundary_head + total_size + termination_tail

    # r1[o][s] <- number of degree-1 check nodes at PD step s at trial o
    r1 = np.zeros((num_repeats, num_pd_steps), dtype='int')
    # rsum[s][pos] = sum of all degrees at position pos at PD step s
    rsum = np.zeros((num_pd_steps, num_positions), dtype='int')
    plrs = np.zeros(num_repeats)

    for o in trange(num_repeats):
        # r[s][pos] <- degree of position pos at step s at the current trial
        r = np.zeros((num_pd_steps, num_positions), dtype='int')
        schedule = empty_schedule()
        uid = 0
        total_generated = 0
        total_recovered = 0

        # 1. Fill in the schedule (step 0)
        for t in range(boundary_head + total_size):
            # num_newborns = gen_poisson_arrival(lam / k)
            num_newborns = 1 if np.random.random() <= (lam / k) else 0
            total_generated += num_newborns
            newborns = []
            for i in range(num_newborns):
                # newborn = gen_user(uid, t, k, vn_rv, frame_len, is_first_fixed, is_sync)
                newborn = gen_user_sc_csa(uid, t, 3, 3, frame_len)
                newborns.append(newborn)
                uid += 1
            add_to_schedule(schedule, newborns)

        # 2. Fill the degrees table from the schedule
        for t in range(num_positions):  # possible termination taken into account
            r[0, t] = len(schedule[t]) if t in schedule else 0
        r[0, 0:boundary_head] = 0
        r1[o, 0] = np.count_nonzero(r[0] == 1)

        # 3. Decode the schedule by picking random degree-1 CN at each step,
        #    update the statistics of the degree distributions.
        stop = total_size
        for l in range(total_size):  # steps of the peeling decoder
            r[l + 1, :] = r[l]  # we start by copying the degrees, will modify those that were changed later
            m = pick_random_deg_1_cn(r[l])  # returns index of a randomly chosen degree-1 cn or None
            if m is None:
                stop = min(stop, l)
                r1[o, l + 1] = r1[o, l]
            else:
                user = head(schedule[m])
                total_recovered += 1
                for s in user.transmissions:
                    schedule[s].remove(user)
                    if len(schedule[s]) == 0:
                        schedule.pop(s)
                    if boundary_head <= s < num_positions: # in terminated case, should always pass
                        r[l + 1, s] -= 1
                        assert r[l + 1, s] >= 0
                    elif boundary_head <= s:
                        # print(s, frame_len, num_positions)
                        assert not is_terminated
                r1[o, l + 1] = np.count_nonzero(r[l + 1] == 1)
                # print(l+1, r1[o,l+1])
        plrs[o] = (total_generated - total_recovered) / total_generated
        print(o, plrs[o], total_generated, total_recovered, total_generated - total_recovered)
        sys.stdout.flush()

        # if r1[o, l + 1] == 1:
        #     print("OMGLOL")

        # rsum += r
        # if len(schedule) == 0:
        #     print("SUCCESS")
        # else: print("FUCKUP")
        # lost_prev, fers_prev = clean(schedule, total_size + frame_len + 1, 0, frame_len, k, is_report)
        # print(o, "Num users:", total_generated, " Steps performed:", min(l, stop), " LOST:", lost_prev)
        # if np.any(r[-1]):
        # print("FAILURE")
        # else: print("SUCCESS")
        # plt.plot(np.arange(total_size + 1), r1)
        # plt.semilogy(np.arange(total_size + 1), r1)
    # plt.ylim(ymin=0)
    # plt.axvline(x=total_size * lam)
    # plt.xlim(xmin=0,xmax=total_size)
    # plt.minorticks_on()
    # plt.grid(which='both')
    # plt.show()
    # rsh = np.reshape(rsum, (rsum.shape[0], -1, frame_len))
    # rns = np.sum(rsh, axis=2)
    # print(rns / num_repeats)
    # with open('r1_3_600_084_fa_csa_trajectories_bounded.pkl', 'wb') as f:
    #     pickle.dump(r1, f)

    return r, r1, plrs


# noinspection SpellCheckingInspection
def simulate_sc_csa(k, lam, vn_deg, vn_pr, is_first_fixed, frame_len, _max_time, is_sync, is_report, total_size):
    vn_rv = stats.rv_discrete(name='vn_deg', values=(vn_deg, vn_pr))

    is_terminated = False
    # users that arrive first will transmit in next frame, so there is a delay by 1
    termination_tail = 1 if not is_terminated else frame_len * 3

    num_repeats = int(1e4)
    max_fuckups = 500
    # num_repeats = 100
    num_fuckups = 0
    num_fuckups_truncated = 0

    total_generated = 0
    total_failed = 0
    total_failed_expurgated = 0

    print(lam, file=sys.stderr)
    the_range = trange(num_repeats)
    for o in the_range:
        schedule = empty_schedule()
        uid = 0

        curr_generated = 0

        for t in range(total_size):
            # num_newborns = gen_poisson_arrival(lam / k)
            num_newborns = 1 if np.random.random() <= (lam / k) else 0
            total_generated += num_newborns
            curr_generated += num_newborns
            newborns = []
            for i in range(num_newborns):
                # newborn = gen_user(uid, t, k, vn_rv, frame_len, is_first_fixed, is_sync)
                newborn = gen_user_sc_csa(uid, t, 3, 3, frame_len)
                newborns.append(newborn)
                uid += 1
            add_to_schedule(schedule, newborns)
            sic_round(schedule, t)
        for t in range(total_size, total_size + termination_tail):
            sic_round(schedule, t)
        # if len(schedule) > 7:
        #     num_fuckups_truncated += 1
        lost = set()
        # old_slots = [ k for k in schedule.keys() if k < total_size - frame_len * 10]
        old_slots = [ k for k in schedule.keys() if k < total_size ]
        for old_slot in old_slots:
            lost |= schedule.pop(old_slot)
        lost = { u for u in lost if all([tr < total_size for tr in u.transmissions]) }

        if len(lost) >= 1:
            num_fuckups += 1

        num_lost = len(lost)
        plr_curr = num_lost / curr_generated

        total_failed += num_lost

        ssets = extract_stopping_sets(lost)
        if any([len(sset) > 2 for sset in ssets]):
            num_fuckups_truncated += 1

        num_lost_exp = sum([len(sset) if len(sset) > 2 else 0 for sset in ssets])
        assert sum([len(sset) for sset in ssets]) == num_lost
        plr_curr_exp = num_lost_exp / curr_generated

        total_failed_expurgated += num_lost_exp

        the_range.set_description("BlockER: %.5f (%.5f); PLR: %.5f (%.5f)" % (num_fuckups / (o + 1), num_fuckups_truncated / (o + 1), plr_curr, plr_curr_exp))
        if num_fuckups >= max_fuckups:
            break

    total_plr = total_failed / total_generated
    total_plr_expurgated = total_failed_expurgated / total_generated

    return (num_fuckups / (o + 1), num_fuckups_truncated / (o + 1), total_plr, total_plr_expurgated)


# noinspection SpellCheckingInspection
def simulate_sc_ldpc(e, l, r, L, M, is_terminated, is_protograph, is_bounded, is_tail_biting, num_repeats=int(1e5), max_fuckups=2000, doping_points=[]):
    is_soft = isinstance(doping_points, dict)
    num_fuckups = 0
    num_fuckups_truncated = 0

    total_generated = 0
    total_failed = 0
    total_failed_expurgated = 0

    total_blocks_generated = 0
    total_blocks_failed_exp = 0
    total_bler_expurgated = 0

    ignored_head = 0 if is_bounded else 20
    ignored_head_schedule = 0 if is_bounded else 10
    ignored_tail = 0 if is_terminated else 20
    L = L + ignored_head + ignored_tail

    cns_per_pos = int(l / r * M)
    num_positions = L + l - 1 if is_terminated else L
    total_size = cns_per_pos * num_positions

    num_doping_points = len(doping_points)

    failures = np.zeros(num_repeats)
    gens = np.zeros(num_repeats)

    if is_protograph:
        if num_doping_points > 0:
            generate_users = lambda: gen_users_sc_ldpc_protograph_doping(l, r, L, M, doping_points, is_tail_biting, e)
        else:
            generate_users = lambda: gen_users_sc_ldpc_protograph(l, r, L, M, is_tail_biting, e)
    else:
        if num_doping_points > 0:
            generate_users = lambda: gen_users_sc_ldpc_doping(l, r, L, M, doping_points, is_tail_biting, e)
        else:
            generate_users = lambda: gen_users_sc_ldpc(l, r, L, M, is_tail_biting, e)

    #print(e, file=sys.stderr)
    the_range = trange(num_repeats)
    #the_range = range(num_repeats)
    for o in the_range:
        schedule = empty_schedule()

        if is_soft:
            curr_generated = (L - ignored_head - ignored_tail) * M
            for vn_position in doping_points:
                num_fixed = int(doping_points[vn_position] * M)
                curr_generated -= num_fixed
            curr_eff_generated = curr_generated  # TODO: Why do I need both?
        else:
            curr_generated = (L - num_doping_points - ignored_head - ignored_tail) * M
            #total_generated += curr_generated
            # Do not consider users at the tail, because we disregard them
            curr_eff_generated = (L - num_doping_points - ignored_head - ignored_tail) * M
        total_generated += curr_eff_generated

        total_blocks_generated += L - num_doping_points - ignored_head - ignored_tail

        gens_curr = 0
        for user in generate_users():
            gens_curr += 1
            add_to_schedule(schedule, [user])
        #gens[o] = gens_curr

        for t in range(ignored_head_schedule * cns_per_pos, total_size):
            sic_round(schedule, t)

        lost = set()
        # Disregard the slots at the tail
        old_slots = [ k for k in schedule.keys() if cns_per_pos * ignored_head <= k < total_size - cns_per_pos * ignored_tail]
        #old_slots = [ k for k in schedule.keys() if k < total_size - cns_per_pos * ignored_tail]
        #old_slots = [ k for k in schedule.keys() if k < total_size ]
        for old_slot in old_slots:
            lost |= schedule.pop(old_slot)
        lost = { u for u in lost if all([tr < total_size for tr in u.transmissions]) }

        if len(lost) >= 1:
            num_fuckups += 1

        num_lost = len(lost)
        plr_curr = num_lost / curr_generated

        total_failed += num_lost
        #failures[o] = num_lost

        ssets = extract_stopping_sets(lost)
        if any([len(sset) > 2 for sset in ssets]):
            num_fuckups_truncated += 1

        num_lost_exp = sum([len(sset) if len(sset) > 2 else 0 for sset in ssets])
        assert sum([len(sset) for sset in ssets]) == num_lost
        plr_curr_exp = num_lost_exp / curr_generated

        lost_exp = set()
        for sset in ssets:
            if len(sset) > 2:
                lost_exp |= { int(u.birthday / cns_per_pos) for u in sset }
        total_blocks_failed_exp += len(lost_exp)

        total_failed_expurgated += num_lost_exp

        total_plr = total_failed / total_generated
        total_plr_expurgated = total_failed_expurgated / total_generated
        total_bler_expurgated = total_blocks_failed_exp / total_blocks_generated

        the_range.set_description("FER: %.5f (%.5f); PLR: %.5f (%.5f); BLER: %.5f" % (num_fuckups / (o + 1), num_fuckups_truncated / (o + 1), total_plr, total_plr_expurgated, total_bler_expurgated))
        if num_fuckups >= max_fuckups:
            break

    return (num_fuckups / (o + 1), num_fuckups_truncated / (o + 1), total_plr, total_plr_expurgated, num_fuckups_truncated, o+1, total_failed_expurgated, total_generated, failures, gens, total_blocks_failed_exp, total_blocks_generated, total_bler_expurgated)


# noinspection SpellCheckingInspection
def simulate_peeling_decoder_ldpc(e, l_deg, r_deg, L, M, is_terminated, is_protograph, num_repeats = None, doping_points=[]):
    if not num_repeats:
        num_repeats = 100
    # num_repeats = int(1e4)
    # num_repeats = 100
    # num_repeats = 1000
    # num_repeats = 4000

    # XXX: add non-bounded case later
    boundary_head = 0
    is_bounded = True

    cns_per_pos = int(l_deg / r_deg * M)
    num_positions = L + l_deg - 1 if is_terminated else L
    total_size = cns_per_pos * num_positions

    num_pd_steps = int(M * num_positions * (e + 0.1))

    num_doping_points = len(doping_points)

    # r1[o][s] <- number of degree-1 check nodes at PD step s at trial o
    r1 = np.zeros((num_repeats, num_pd_steps + 1), dtype='int')
    plrs = np.zeros(num_repeats)

    if is_protograph:
        if num_doping_points > 0:
            generate_users = lambda: gen_users_sc_ldpc_protograph_doping(l_deg, r_deg, L, M, doping_points, False, e)
        else:
            generate_users = lambda: gen_users_sc_ldpc_protograph(l_deg, r_deg, L, M, False, e)
    else:
        if num_doping_points > 0:
            generate_users = lambda: gen_users_sc_ldpc_doping(l_deg, r_deg, L, M, doping_points, False, e)
        else:
            generate_users = lambda: gen_users_sc_ldpc(l_deg, r_deg, L, M, False, e)

    for o in trange(num_repeats):
    #for o in range(num_repeats):
        r = np.zeros(total_size, dtype='int')
        schedule = empty_schedule()

        total_generated = 0

        curr_generated = (L - num_doping_points) * M
        total_generated += curr_generated

        for user in generate_users():
            add_to_schedule(schedule, [user])

        total_recovered = total_generated - user.uid - 1

        # 2. Fill the degrees table from the schedule
        for t in range(total_size):  # possible termination taken into account
            r[t] = len(schedule[t]) if t in schedule else 0
        r1[o, 0] = np.count_nonzero(r == 1)

        # 3. Decode the schedule by picking random degree-1 CN at each step,
        #    update the statistics of the degree distributions.
        stop = num_pd_steps
        for l in range(num_pd_steps):  # steps of the peeling decoder
            m = pick_random_deg_1_cn(r)  # returns index of a randomly chosen degree-1 cn or None
            if m is None:
                stop = min(stop, l)
                r1[o, l + 1] = r1[o, l]
            else:
                user = head(schedule[m])
                total_recovered += 1
                for s in user.transmissions:
                    schedule[s].remove(user)
                    if len(schedule[s]) == 0:
                        schedule.pop(s)
                    if boundary_head <= s < total_size: # in terminated case, should always pass
                        r[s] -= 1
                        assert r[s] >= 0
                    elif boundary_head <= s:
                        # print(s, M, total_size)
                        assert not is_terminated
                r1[o, l + 1] = np.count_nonzero(r == 1)
                # print(l+1, r1[o,l+1])
        if r1[o, l+1] == 1:
            print("The number of degree-1 CNs at the end of the iterations is 1!!!")
        plrs[o] = (total_generated - total_recovered) / total_generated
        #print(o, plrs[o], total_generated, total_recovered, total_generated - total_recovered)
        #sys.stdout.flush()

    return None, r1, plrs


# noinspection SpellCheckingInspection
def simulate_peeling_decoder_ldpc_uncoupled(e, l_deg, r_deg, M, num_repeats = None):
    if not num_repeats:
        num_repeats = 100
    # num_repeats = int(1e4)
    # num_repeats = 100
    # num_repeats = 1000
    # num_repeats = 4000

    cns_per_pos = int(l_deg / r_deg * M)
    total_size = cns_per_pos

    num_pd_steps = int(M * (e + 0.1))

    # r1[o][s] <- number of degree-1 check nodes at PD step s at trial o
    r1 = np.zeros((num_repeats, num_pd_steps + 1), dtype='int')
    plrs = np.zeros(num_repeats)

    generate_users = lambda: gen_users_ldpc(l_deg, r_deg, M, e)

    num_vns_lst = []
    for o in trange(num_repeats):
    #for o in range(num_repeats):
        r = np.zeros(total_size, dtype='int')
        schedule = empty_schedule()

        total_generated = 0

        curr_generated = M
        total_generated += curr_generated

        num_vns = 0
        for user in generate_users():
            add_to_schedule(schedule, [user])
            num_vns += 1
        num_vns_lst.append(num_vns)

        total_recovered = total_generated - user.uid - 1

        # 2. Fill the degrees table from the schedule
        for t in range(total_size):  # possible termination taken into account
            r[t] = len(schedule[t]) if t in schedule else 0
        r1[o, 0] = np.count_nonzero(r == 1)

        # 3. Decode the schedule by picking random degree-1 CN at each step,
        #    update the statistics of the degree distributions.
        stop = num_pd_steps
        for l in range(num_pd_steps):  # steps of the peeling decoder
            m = pick_random_deg_1_cn(r)  # returns index of a randomly chosen degree-1 cn or None
            if m is None:
                stop = min(stop, l)
                r1[o, l + 1] = r1[o, l]
            else:
                user = head(schedule[m])
                total_recovered += 1
                for s in user.transmissions:
                    if s in schedule:
                        if user in schedule[s]:
                            schedule[s].remove(user)
                        else:
                            print("no user in schedule[s]: ", user)
                            sys.exit(-1)
                    else:
                        print("no schedule[s] for user: ", user)
                        sys.exit(-1)
                    if len(schedule[s]) == 0:
                        schedule.pop(s)
                    r[s] -= 1
                    assert r[s] >= 0
                r1[o, l + 1] = np.count_nonzero(r == 1)
                # print(l+1, r1[o,l+1])
        if r1[o, l+1] == 1:
            print("The number of degree-1 CNs at the end of the iterations is 1!!!")
        plrs[o] = (total_generated - total_recovered) / total_generated
        #print(o, plrs[o], total_generated, total_recovered, total_generated - total_recovered)
        #sys.stdout.flush()

    return None, r1, plrs, num_vns_lst


def simulate_pd_covariance(e, l_deg, r_deg, L, M, is_terminated, is_protograph, steps, num_repeats=None):
    if not num_repeats:
        num_repeats = 100
    # num_repeats = int(1e4)
    # num_repeats = 100
    # num_repeats = 1000
    # num_repeats = 4000

    # XXX: add non-bounded case later
    boundary_head = 0
    is_bounded = True

    cns_per_pos = int(l_deg / r_deg * M)
    num_positions = L + l_deg - 1 if is_terminated else L
    total_size = cns_per_pos * num_positions

    num_pd_steps = int(M * num_positions * (e + 0.1))

    # r1[o][s] <- number of degree-1 check nodes at PD step s at trial o
    r1 = np.zeros((num_repeats, num_pd_steps + 1), dtype='int')
    plrs = np.zeros(num_repeats)

    ctx_dict = {}
    g_dict = {}
    for stp in steps:
        ctx_dict[stp] = init_cov(num_positions * (r_deg + 1))
    max_stp = max(steps)

    if is_protograph:
        generate_users = lambda: gen_users_sc_ldpc_protograph(l_deg, r_deg, L, M, e)
    else:
        generate_users = lambda: gen_users_sc_ldpc(l_deg, r_deg, L, M, e)

    for o in trange(num_repeats):
    #for o in range(num_repeats):
        r = np.zeros(total_size, dtype='int')
        schedule = empty_schedule()

        total_generated = 0

        curr_generated = L * M
        total_generated += curr_generated

        for stp in steps:
            g_dict[stp] = np.zeros((num_positions, r_deg + 1), dtype='int')

        for user in generate_users():
            if -1 in steps:
                g = g_dict[-1]
                g[int(user.birthday / cns_per_pos), 0] += 1
            add_to_schedule(schedule, [user])

        total_recovered = total_generated - user.uid - 1

        # 2. Fill the degrees table from the schedule
        for t in range(total_size):  # possible termination taken into account
            r[t] = len(schedule[t]) if t in schedule else 0
        r1[o, 0] = np.count_nonzero(r == 1)

        if -1 in steps:
            g = g_dict[-1]
            for pos in range(num_positions):
                for cnpos in range(cns_per_pos):
                    t = pos * cns_per_pos + cnpos
                    deg = r[t]
                    if deg != 0:
                        g[pos, deg] += deg
            ctx = ctx_dict[-1]
            ctx = update_cov(np.ravel(g), ctx)
            ctx_dict[-1] = ctx

            if is_terminated and np.sum(g[:,1:]) != np.sum(g[:,0]) * l_deg:
                print(g)
                print("INCONSISTENCY! The number of edges does not match the number of VNs (TERM ONLY)!")
                sys.exit(-1)

        if max_stp == -1:
            continue

        # 3. Decode the schedule by picking random degree-1 CN at each step,
        #    update the statistics of the degree distributions.
        stop = num_pd_steps
        for l in range(num_pd_steps):  # steps of the peeling decoder
            m = pick_random_deg_1_cn(r)  # returns index of a randomly chosen degree-1 cn or None
            if m is None:
                stop = min(stop, l)
                r1[o, l + 1] = r1[o, l]
            else:
                user = head(schedule[m])
                total_recovered += 1
                for s in user.transmissions:
                    schedule[s].remove(user)
                    if len(schedule[s]) == 0:
                        schedule.pop(s)
                    if boundary_head <= s < total_size: # in terminated case, should always pass
                        r[s] -= 1
                        assert r[s] >= 0
                    elif boundary_head <= s:
                        # print(s, M, total_size)
                        assert not is_terminated
                r1[o, l + 1] = np.count_nonzero(r == 1)
                # print(l+1, r1[o,l+1])

            # Update the covariance for the current step (if it is among the steps of interest)
            if l in steps:
                g = g_dict[l]
                usrs = set()
                for pos in range(num_positions):
                    for cnpos in range(cns_per_pos):
                        t = pos * cns_per_pos + cnpos
                        deg = r[t]
                        if deg != 0:
                            g[pos, deg] += deg
                        if t in schedule:
                            slt = schedule[t]
                            for u in slt:
                                usrs.add(u)
                for user in usrs:
                    g[int(user.birthday / cns_per_pos), 0] += 1
                ctx = ctx_dict[l]
                ctx = update_cov(np.ravel(g), ctx)
                ctx_dict[l] = ctx
                if l == max_stp:
                    continue

        if r1[o, l+1] == 1:
            print("The number of degree-1 CNs at the end of the iterations is 1!!!")
        plrs[o] = (total_generated - total_recovered) / total_generated
        #print(o, plrs[o], total_generated, total_recovered, total_generated - total_recovered)
        #sys.stdout.flush()

    for stp, ctx in ctx_dict.items():
        cov = get_cov(ctx)
        _, means, _ = ctx
        #print(cov)
        #print(means)
        print("stp = {stp}".format(**locals()))
        for u in range(num_positions):
            for x in range(num_positions):
                print("u=%d, x=%d" % (u, x))
                for j in range(r_deg + 1):
                    for z in range(r_deg + 1):
                        print("%9.6f" % cov[u * (r_deg + 1) + j][x * (r_deg + 1) + z], end=' ')
                    print()
                print()
        print()

    return None, r1, plrs


def pick_random_deg_1_cn(a):
    indices = np.flatnonzero(a == 1)
    if len(indices) == 0:
        return None
    return random.choice(indices)


def calc_plr(lost, gen):
    if gen == 0:
        return 0
    return lost / gen


def calc_fer(frames_lost, t, clean_window, frame_len, k):
    assert clean_window % frame_len == 0
    flen = frame_len * k
    t_clean = t - clean_window
    total_frames = (t_clean // flen) + 1
    return frames_lost / total_frames


def report(schedule, t, total_generated, total_decoded, lost_old, generated_old):
    schedule_size = len(schedule)
    print('{:<12} {:>12} {:>12} {:>12} {:>12} {:>12.8f}'.format(t, schedule_size, total_generated, total_decoded,
                                                                lost_old, calc_plr(lost_old, generated_old)))


def clean(schedule, t, clean_window, frame_len, k, is_report):
    lost = set()
    old_slots = [k for k in schedule.keys() if k < t - clean_window]
    for old_slot in old_slots:
        lost |= schedule.pop(old_slot)
    for u in lost:
        for idx in u.transmissions:
            if idx in schedule:
                schedule[idx].remove(u)  # important for FA
    frames = set([frame_sync_offset(u.birthday, frame_len * k) // (frame_len * k) for u in lost])
    l_ = len(lost)
    f = len(frames)
    if is_report:
        report_ssets(lost)
    return l_, f


def report_ssets(lost):
    ssets = extract_stopping_sets(lost)
    for sset in ssets:
        print("SS,", len(sset))
        s = empty_schedule()
        add_to_schedule(s, sset)
        print(sorted(s.keys()))
        for u in sset:
            print(u.uid, sorted(u.transmissions), u.recovered)


def extract_stopping_sets(userset):
    s = empty_schedule()
    add_to_schedule(s, userset)
    ssets = []
    while userset:
        u = userset.pop()
        sset = {u}
        induced = {u}
        # print(userset, u, s)
        while induced:
            usr = induced.pop()
            for t in usr.transmissions:
                slot = s[t]
                induced |= slot - sset
                sset |= slot
        userset.difference_update(sset)
        # normalize_stopping_set(sset)
        ssets.append(sset)
    return ssets


def normalize_stopping_set(sset):
    slot_dict = {}
    j = 0
    for u in sset:
        k = 0
        for t in u.transmissions:
            if t not in slot_dict:
                ind = j
                slot_dict[t] = j
                j += 1
            else:
                ind = slot_dict[t]
            u.transmissions[k] = ind
            k += 1

        for t in list(u.recovered):
            if t not in slot_dict:
                ind = j
                slot_dict[t] = j
                j += 1
            else:
                ind = slot_dict[t]
            u.recovered.remove(t)
            u.recovered.add(ind)

    return sset


def simulate(fname, is_first_fixed, is_sync, k, frame_len, max_time, lams, is_report):
    vn_deg = [k * 3]  # , k * 8]
    vn_pr = [1.0]  # [0.86, 0.14]
    # vn_deg = [6, 5]
    # vn_pr = [0.3, 0.7]
    # vn_deg = [2, 3, 8]
    # vn_pr = [0.5, 0.28, 0.22]
    plrs = np.zeros(len(lams))
    fers = np.zeros(len(lams))
    i = 0
    for lam in lams:
        plr, fer = simulate_fa_csa(k, lam, vn_deg, vn_pr, is_first_fixed, frame_len, max_time, is_sync, is_report)
        plrs[i] = plr
        fers[i] = fer
        print("SIMULATION POINT. lam =", lam, "plr =", plr, "fer = ", fer)
        print(lam, plr, fer, file=sys.stderr)
        if plr < 1e-8:
            break
        i += 1
    if fname != 'none':
        scipy.io.savemat(fname, {fname + '_lams': lams, fname + '_plrs': plrs, fname + '_fers': fers})


def test_simulation():
    k = 2  # IRSA
    lam = 0.8
    # This is the degree distribution of n-s
    vn_deg = [6, 16]
    vn_pr = [0.86, 0.14]
    frame_len = 200
    max_time = int(1e8)
    is_first_fixed = False
    is_sync = False
    is_report = True
    plr_ = simulate_fa_csa(k, lam, vn_deg, vn_pr, is_first_fixed, frame_len, max_time, is_sync, is_report)
    print(plr_)


def test_2_6_csa_sync():
    u1 = User(1, 0, [1, 2, 3, 4, 7, 10], set(), 2)
    u2 = User(2, 0, [1, 2, 3, 5, 6, 9], set(), 2)
    u3 = User(3, 0, [2, 3, 4, 5, 6, 8], set(), 2)

    newborns = [u1, u2, u3]
    schedule = empty_schedule()
    add_to_schedule(schedule, newborns)
    for t in range(11):
        decoded = sic_round(schedule, t)
        print(t, decoded)
    print(schedule)


def test_unif():
    for i in range(10):
        print(unif())


def test_peeling_decoder():
    # chain_len = 200 + 10
    chain_len = 200
    # chain_len = 100
    frame_len = 500
    # frame_len = 200
    # frame_len = 8000
    # frame_len = 1000
    # frame_len = 3

    # r, r1, plrs = simulate_peeling_decoder(1, 0.86, [3], [1.0], False, frame_len, None, False, False, frame_len * chain_len)
    # with open('r1_sc_csa_1000_086_100_nonterminated.pkl', 'wb') as f:
    #     pickle.dump((r1, plrs), f)
    # for g in [0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9]:
    # for g in [0.84, 0.845, 0.85, 0.855, 0.86, 0.865, 0.87, 0.875, 0.88, 0.885, 0.89, 0.895, 0.9]:
    # for g in np.arange(0.92, 0.89, -0.0025):
    # for g in np.arange(0.88, 0.84, -0.0025):
    for g in [0.85]:
    # for g in np.arange(0.86, 0.80, -0.0025):
        ber, ber_truncated, plr, plr_exp = simulate_sc_csa(1, g, [3], [1.0], False, frame_len, None, False, False, frame_len * chain_len)
        # print(g, ber, ber_truncated, file=sys.stderr)
        print(g, ber, ber_truncated, plr, plr_exp)
        sys.stdout.flush()

    # return simulate_sc_csa(1, 0.84, [3], [1.0], False, frame_len, None, False, False, frame_len * 100)
    # return simulate_peeling_decoder(1, 0.84, [3], [1.0], False, frame_len, None, False, False, frame_len * 100)
    # return simulate_peeling_decoder(1, 0.84, [3], [1.0], False, 600, None, False, False, 200 * 100)
    # return simulate_peeling_decoder(1, 0.8, [2], [1.0], False, frame_len, None, False, False, frame_len * 50)

def test_sc_ldpc():
    l = 4
    r = 8
    L = 50
    M = 10000
    # es = np.arange(0.475, 0.46, -0.0005)
    # es = np.arange(0.485, 0.47, -0.0005)
    # es = np.arange(0.475, 0.45, -0.005)
    # es = np.arange(0.47, 0.4495, -0.0005)
    # es = np.arange(0.48, 0.46, -0.0005)
    # es = [0.4]
    is_terminated = False
    # is_terminated = False
    # is_protograph = True
    is_protograph = False

    # for e in es:
    #     ber, ber_truncated, plr, plr_exp, fbl, tbl, fbit, tgen = simulate_sc_ldpc(e, l, r, L, M, is_terminated, is_protograph, is_bounded)
    #     print(e, ber, ber_truncated, plr, plr_exp)
    #     sys.stdout.flush()

    e = 0.48
    doping_points = []
    # L = 100  # needs to be equal to the true length in the current implementation
    # doping_points = [50,51,52]
    # L += len(doping_points)
    num_runs = 100
    num_runs_batch = 100
    num_rounds = int(num_runs / num_runs_batch)
    for i in range(num_rounds):
        _, r1, plrs = simulate_peeling_decoder_ldpc(e, l, r, L, M, is_terminated, is_protograph, num_runs_batch, doping_points)
        #with open(f"r1_sc_ldpc_5_10_100_10000_0478_terminated_doped_50_51_52_{i}.pkl", 'wb') as f:
        with open(f"../../sim_data/trajectories_peeling_decoding/r1_sc_ldpc_4_8_50_10000_048_nonterminated_{i}.pkl", 'wb') as f:
            pickle.dump((r1, plrs), f)

def test_ldpc():
    l = 3
    r = 6
    M = 10000

    e = 0.39

    num_runs = 1000
    num_runs_batch = 1000
    num_rounds = int(num_runs / num_runs_batch)

    for i in range(num_rounds):
        _, r1, plrs, num_vns = simulate_peeling_decoder_ldpc_uncoupled(e, l, r, M, num_runs_batch)
        with open(f"r1_ldpc_3_6_10000_039_v3_{i}.pkl", 'wb') as f:
            pickle.dump((r1, plrs, num_vns), f)


def main_simulate_variance():
    fname = sys.argv[1]
    l = int(sys.argv[2])
    r = int(sys.argv[3])
    L = int(sys.argv[4])
    M = int(sys.argv[5])
    e = float(sys.argv[6])
    is_terminated = True if sys.argv[7] == 'T' else False
    is_protograph = True if sys.argv[8] == 'P' else False
    num_runs = int(sys.argv[9])
    num_runs_batch = int(sys.argv[10])
    ftheory = sys.argv[11]


    with open(ftheory, 'rb') as f:
        r1s_theory = pickle.load(f)[0]

    lvl, idxs = find_level(r1s_theory)

    isfirst = True
    ssquares, counts = None, None

    num_rounds = int(num_runs / num_runs_batch)
    for i in range(num_rounds):
        _, r1s, _ = simulate_peeling_decoder_ldpc(e, l, r, L, M, is_terminated, is_protograph, num_runs_batch)
        ssquares_chunk, counts_chunk = calc_nu_chunk(r1s, r1s_theory, M)
        ssquares = ssquares_chunk if isfirst else ssquares + ssquares_chunk
        counts = counts_chunk if isfirst else counts + counts_chunk
        isfirst = False
    with open(fname, 'wb') as f:
        pickle.dump((ssquares, counts), f)


def simulate_covariance():
    l = 3
    r = 6
    L = 20
    M = 1000
    e = 0.47
    is_terminated = False
    is_protograph = False
    #steps = np.array([0])
    steps = np.array([-1, 20])
    num_repeats = int(1e5)
    print("({l},{r},{M},{L}) terminated={is_terminated} over e={e} and steps={steps}".format(**locals()))
    simulate_pd_covariance(e, l, r, L, M, is_terminated, is_protograph, steps, num_repeats)


def main():
    # test()
    # test_simulation()
    # print(frame_sync_offset(259, 200))
    # main_simulate()
    # test_2_6_csa_sync()
    # test_unif()
    # test_peeling_decoder()
    test_sc_ldpc()
    # test_ldpc()
    # main_simulate_sc_ldpc()
    # main_simulate_variance()
    # simulate_covariance()


def main_simulate_sc_ldpc():
    fname = sys.argv[1]
    l = int(sys.argv[2])
    r = int(sys.argv[3])
    L = int(sys.argv[4])
    M = int(sys.argv[5])
    es = eval(sys.argv[6])
    is_terminated = True if sys.argv[7] == 'T' else False
    is_protograph = True if sys.argv[8] == 'P' else False
    is_bounded = True if sys.argv[9] == 'B' else False
    is_tail_biting = True if sys.argv[10] == 'TB' else False
    num_repeats = int(sys.argv[11])
    max_fuckups = int(sys.argv[12])
    doping_points = eval(sys.argv[13])
    # failures_fname = sys.argv[14]
    # I decided to artificially do that, otherwise I might one day forget to do that in the calling script
    L += len(doping_points)

    with open(fname, 'wt') as f:
        print(f"# SC-LDPC ({l},{r},L={L},M={M}) terminated:{is_terminated}, proto:{is_protograph}, bounded:{is_bounded}, tail biting:{is_tail_biting}. num_repeats={num_repeats}, max_fuckups={max_fuckups}, doping_points={doping_points}.")
        print(f"# SC-LDPC ({l},{r},L={L},M={M}) terminated:{is_terminated}, proto:{is_protograph}, bounded:{is_bounded}, tail biting:{is_tail_biting}. num_repeats={num_repeats}, max_fuckups={max_fuckups}, doping_points={doping_points}.", file=f)
        f.flush()
        for e in es:
            ber, ber_truncated, plr, plr_exp, fbl, tbl, fbit, tgen, _flrs, _gens, fblocks, tblocks, bler = simulate_sc_ldpc(e, l, r, L, M, is_terminated, is_protograph, is_bounded, is_tail_biting, num_repeats, max_fuckups, doping_points)
            print(e, ber, ber_truncated, plr, plr_exp, fbl, tbl, fbit, tgen, fblocks, tblocks, bler, file=f)
            print(e, ber, ber_truncated, plr, plr_exp, fbl, tbl, fbit, tgen, fblocks, tblocks, bler)
            f.flush()
            sys.stdout.flush()
    # with open(failures_fname, 'wb') as f:
    #     pickle.dump((flrs, gens), f)


def main_simulate():
    fname = sys.argv[1]
    is_first_fixed = True if sys.argv[2] == 'F' else False
    is_sync = False if not sys.argv[3] or sys.argv[3] != 'SYNC' else True
    k = 1 if not sys.argv[4] else int(sys.argv[4])
    flen = 200 if not sys.argv[5] else int(sys.argv[5])
    max_time = int(1e7) if not sys.argv[6] else int(eval(sys.argv[6]))
    lams = [0.36] if len(sys.argv) <= 7 else eval(sys.argv[7])
    print(lams)
    is_report = False if len(sys.argv) <= 8 or sys.argv[8] != 'REPORT' else True
    simulate(fname, is_first_fixed, is_sync, k, flen, max_time, lams, is_report)


if __name__ == '__main__':
    main()
