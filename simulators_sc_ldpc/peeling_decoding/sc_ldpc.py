import numpy as np


def perm(k):
    return np.random.permutation(k)


def inv_perm(p):
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def permute(word, perm):
    return word[perm]


def group(a, num_groups):
    return np.reshape(a, (num_groups, -1))


def gen_cn_profile(l, M):
    num_sockets = l * M
    a_perm = perm(num_sockets)
    perm_groups = group(a_perm, l)
    return perm_groups


def profile_to_cn_indices(profile, r):
    return profile // r


def gen_vn_indices(l, r, L, M):
    num_cns = int(l * M / r)
    D = L + l - 1
    cn_indices = np.array([i * num_cns + profile_to_cn_indices(gen_cn_profile(l, M), r) for i in range(D)])
    vn_indices = np.array([[cn_indices[i + d][d] for d in range(l)] for i in range(L)])
    return vn_indices


def gen_vn_indices_tail_biting(l, r, L, M):
    num_cns = int(l * M / r)
    cn_indices = np.array([i * num_cns + profile_to_cn_indices(gen_cn_profile(l, M), r) for i in range(L)])
    vn_indices = np.array([[cn_indices[(i + d) % L][d] for d in range(l)] for i in range(L)])
    return vn_indices


def vn_indices_to_transmissions(vn_indices, l, L, M):
    transmissions = np.array([ [vn_indices[i,k,u] for k in range(l)] for i in range(L) for u in range(M) ])
    return transmissions


def gen_slots(l, r, L, M):
    vn_indices = gen_vn_indices(l, r, L, M)
    transmissions = vn_indices_to_transmissions(vn_indices, l, L, M)
    return transmissions


def gen_slots_tail_biting(l, r, L, M):
    vn_indices = gen_vn_indices_tail_biting(l, r, L, M)
    transmissions = vn_indices_to_transmissions(vn_indices, l, L, M)
    return transmissions


def main():
    l = 3
    r = 6
    M = 4
    L = 2
    vn_indices = gen_vn_indices(l, r, L, M)
    print(vn_indices)

    transmissions = vn_indices_to_transmissions(vn_indices, l, L, M)
    print(transmissions)

    L = 3
    vn_indices_tb = gen_vn_indices_tail_biting(l, r, L, M)
    print(vn_indices_tb)

    transmissions_tb = vn_indices_to_transmissions(vn_indices_tb, l, L, M)
    print(transmissions_tb)

if __name__ == '__main__':
    main()
