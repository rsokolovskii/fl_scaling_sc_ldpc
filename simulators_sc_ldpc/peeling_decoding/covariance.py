import numpy as np


# This is from Wikipedia
# It does what I need, but I need to refurbish it so that it
# 1. Works with matrices (arrays?)
# 2. Can be called one update at a time
def online_covariance(data1, data2):
    meanx = meany = C = n = 0
    for x, y in zip(data1, data2):
        n += 1
        dx = x - meanx
        dy_old = y - meany
        meanx += dx / n
        meany += dy_old / n
        dy = y - meany
        C += dx * dy

    population_covar = C / n
    # Bessel's correction for sample variance
    sample_covar = C / (n - 1)
    return sample_covar


# You give me a 1-D array, I will initialize the covariance matrix
def init_cov(size):
    cov = np.zeros((size, size), dtype='double')
    means = np.zeros(size, dtype='double')
    n = 0
    return cov, means, n


def update_cov(array, ctx):
    cov, means, n = ctx
    n += 1
    dx_old = array - means
    means += dx_old / n
    dx = array - means
    dd = np.outer(dx_old, dx)
    cov += dd
    return cov, means, n


def get_cov(ctx, sample=True):
    cov, _, n = ctx
    if sample:
        return cov / (n - 1)
    return cov / n


def __test_cov_online(a, b):
    array = np.array([a, b]).T
    ctx = init_cov(len(array[0]))
    for i in range(len(array)):
        ctx = update_cov(array[i], ctx)
    return get_cov(ctx)


def __test_cov_online_array(array):
    arr = array[0]
    ctx = init_cov(len(arr))
    for i in range(len(array)):
        ctx = update_cov(array[i], ctx)
    return get_cov(ctx)


def covariance(a, b):
    return np.cov(a, b)


def naive_covariance(a, b):
    ma = np.mean(a)
    mb = np.mean(b)
    return np.mean((a - ma) * (b - mb))


def naive_covariance_v2(a, b):
    ma = np.mean(a)
    mb = np.mean(b)
    return np.mean(a * b) - ma * mb


def test_est_covariance():
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    print(naive_covariance(a,b))
    print(naive_covariance_v2(a,b))
    print(np.cov(np.array([a,b]), bias=True))
    print(online_covariance(a, b))
    print(__test_cov_online(a, b))


def test_est_covariance_2():
    a = np.array([0, 1, 2])
    b = np.array([2, 1, 0])
    print(naive_covariance(a,b))
    print(naive_covariance_v2(a,b))
    print(np.cov(np.array([a,b])))
    print(online_covariance(a, b))
    print(__test_cov_online(a, b))


def test_est_covariance_normal():
    cov = np.array([[1, 0.5, 0.7],
                    [0.5, 4, 1.5],
                    [0.7, 1.5, 1]])
    mean = np.array([0,0,0])
    size = 100000
    arr = np.random.multivariate_normal(mean, cov, size=size)

    print(cov)
    print(np.cov(arr.T))
    print(__test_cov_online_array(arr))


def main():
    #test_est_covariance()
    #test_est_covariance_2()
    test_est_covariance_normal()


if __name__ == '__main__':
    main()
