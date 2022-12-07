import numpy as np

from sc_ldpc import perm, inv_perm, permute, group


def augment_edge(num_cns):
    a_perm = perm(num_cns)
    return a_perm


# each slots portion corresponds to one VN in the protograph
def gen_slots_portion(l, num_cns):
    augmented_edges = np.array([i * num_cns + augment_edge(num_cns) for i in range(l)])
    return augmented_edges.T


def gen_slots_from_position(l, r, M):
    num_cns = int(l * M / r)
    num_portions = int(M / num_cns)
    return np.vstack([ gen_slots_portion(l, num_cns) for _ in range(num_portions) ])


def main():
    l = 3
    r = 6
    M = 10
    L = 2

    slots_from_position = gen_slots_from_position(l, r, M)
    print(slots_from_position)


if __name__ == '__main__':
    main()
