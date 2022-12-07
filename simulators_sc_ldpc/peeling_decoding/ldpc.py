import numpy as np


# buggy (prone to deadlock loop and emergency escape) version
def s_interleaver(k, ld, rd):
    used = set()
    whole = set(np.arange(k))
    perm = np.zeros(k, dtype='int')
    for i in range(k):
        possible = list(whole - used)
        iteration = 0
        while True:
            candidate = np.random.choice(possible)
            distances = np.array([abs(perm[max(0, i - j)] - candidate) for j in np.arange(1, ld)])
            #print(possible, distances)
            iteration += 1
            if all(distances > rd) or iteration > k: break
        used.add(candidate)
        perm[i] = candidate
    return perm


def perm(k):
    # return s_interleaver(k, 3, 6)
    return np.random.permutation(k)


def inv_perm(p):
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def permute(word, perm):
    return word[perm]


def group(a, num_groups):
    return np.reshape(a, (num_groups, -1))


def gen_cn_profile(l, N):
    num_sockets = l * N
    a_perm = perm(num_sockets)
    perm_groups = group(a_perm, l)
    return perm_groups


def profile_to_cn_indices(profile, r):
    return profile // r


def gen_vn_indices(l, r, N):
    return profile_to_cn_indices(gen_cn_profile(l, N), r)


def gen_vn_indices_checked(l, r, N, is_bad = lambda _vn_indices : False):
    vn_indices = gen_vn_indices(l, r, N)
    while is_bad(vn_indices):
        vn_indices = gen_vn_indices(l, r, N)
    return vn_indices


def vn_indices_to_transmissions(vn_indices, l, N):
    transmissions = np.array([ [vn_indices[k,u] for k in range(l)] for u in range(N) ])
    return transmissions


# Does not allow VNs that are connected to the same CN more than once.
def contains_repeats(transmissions):
    return any([ len(np.unique(tr)) != len(tr) for tr in transmissions ])


def gen_slots_unchecked(l, r, N):
    vn_indices = gen_vn_indices(l, r, N)
    transmissions = vn_indices_to_transmissions(vn_indices, l, N)
    return transmissions


def gen_slots(l, r, N, is_bad = contains_repeats):
    transmissions = gen_slots_unchecked(l, r, N)
    while is_bad(transmissions):
        transmissions = gen_slots_unchecked(l, r, N)
    return transmissions


def main():
    l = 3
    r = 6
    N = 10

    transmissions = gen_slots(l, r, N)
    print(transmissions)


if __name__ == '__main__':
    main()
