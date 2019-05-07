import numpy as np
from matplotlib import pyplot as plt
import copy


def get_gammas_px(x, pi, mi, sigma):
    N = x.shape[0]
    K = pi.shape[0]
    gam = np.zeros((N, K))
    for k in range(K):
        tmp1 = 1 / np.sqrt(2 * np.pi * sigma[k])
        tmp2 = np.exp(-np.power((x - mi[k]), 2) / (2 * sigma[k]))
        gam[:, k] = pi[k] * np.prod(tmp1 * tmp2, axis=1)
    px = np.sum(gam, axis=1)
    for k in range(K):
        gam[:, k] = gam[:, k] / px
    return gam, px


def new_mkd(gammas, x):
    K = gammas.shape[1]
    D = x.shape[1]
    mi = np.zeros((K, D))
    sum_g = np.sum(gammas, axis=0)
    for k in range(K):
        for d in range(D):
            mi[k, d] = np.sum(gammas[:, k] * x[:, d], axis=0) / sum_g[k]
    return mi


def new_sk_d(gammas, x, mi):
    K = gammas.shape[1]
    D = x.shape[1]
    sigma = np.zeros((K, 1))
    sum_g = np.sum(gammas, axis=0)
    for k in range(K):
        sigma[k] = np.sum(gammas[:, k] * np.sum(np.power(x - mi[k], 2), axis=1)) / (D * sum_g[k])
    return sigma


def new_pk(gammas):
    N = gammas.shape[0]
    pi = np.zeros((K, 1))
    for k in range(K):
        pi[k] = np.sum(gammas[:, k]) / N
    return pi


def calc_loss(px):
    return np.sum(np.log(px))


def calc_error(x_true, x_r):
    return np.sum((np.linalg.norm(x_true - x_r, axis=1) ** 2)) / x_true.shape[0]


# img = plt.imread('./scarlet_tanager.jpg')
# img = plt.imread('./im.jpg')
img = plt.imread('./acropolis.jpg')

x = np.asarray(img).reshape(-1, img.shape[2]) / 255

x_true = copy.deepcopy(x)

N = len(x)
K = 64
D = 3

thres = 0.1
converged = False

steps = 100

print("Starting Image Segmentation with EM Algorithm")
print("Image Shape:", img.shape)
print("N:", N)
print("K:", K)
print("Steps:", steps)
print("Threshold of convergence:", thres)
print()

# Initialize pi, mi, sigma
pi = np.asarray([1 / K] * K)
mi = np.asarray([np.random.uniform(0.1, 0.9, D) for i in range(K)])
sigma = np.random.uniform(0.2, 0.6, K)

errors = []
logL = []

logL.append(float('-inf'))

for s in range(steps):

    if converged:
        print('@'*17, 'The Log Likelihood has converged', '@'*17)
        break

    print('='*30, 'STEP %s' %s, '='*30)

    # Expectation step
    print('-'*27, "Expectation ", '-'*27)

    gammas, px = get_gammas_px(x, pi, mi, sigma)

    logL.append(calc_loss(px))
    print("\nLog Likelihood: %f\n" % logL[-1])
    #

    # Check that log loss is increasing and check threshold
    logL_diff = logL[-1] - logL[-2]
    if logL_diff < 0:
        print('ERROR: Log Likelihood is not increasing!!')
        exit(1)
    if np.abs(logL_diff) < thres:
        converged = True

    # Maximization step
    print('-' * 27, "Maximization", '-' * 27)
    # print("Assigning new mis...")
    mi = new_mkd(gammas, x)
    # print("Assigning new sigmas...")
    sigma = new_sk_d(gammas, x, mi)
    # print("Assigning new pis...")
    pi = new_pk(gammas)
    #
    # print('\nAnalytics\n')
    # print('mi: %s\nsigma: %s\npi: %s' % (str([e for e in mi]), str([e for e in sigma]), str([e for e in pi])))

    new_x = [[mi[i][0], mi[i][1], mi[i][2]] for i in
             np.argmax(np.asarray(gammas), axis=1)]

    errors.append(calc_error(x_true, new_x))
    print("\nError: %f\n" % errors[-1])

    new_img = np.asarray(new_x).reshape(img.shape[0], img.shape[1], img.shape[2])
    plt.imshow(new_img)
    plt.show()
