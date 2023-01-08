import numpy as np

data = [[], [], []]
data[0] = np.loadtxt('data1')
data[1] = np.loadtxt('data2')
data[2] = np.loadtxt('data3')

weights = [0.341398243018411, 0.367330235091507, 0.291271521890082]
means = [
    [3.006132088737974, 3.093100568285389],
    [0.196675859954268, -0.034521603109466],
    [-2.957520528756456, 2.991192198151507]
]
covariances = np.zeros((2, 2, 3))
covariances[:, :, 0] = [
    [0.949104844872119, -0.170637132238246],
    [-0.170637132238246, 2.011158266600814]
]
covariances[:, :, 1] = [
    [0.837094104536474, 0.044657749659523],
    [0.044657749659523, 1.327399518241827]
]
covariances[:, :, 2] = [
    [1.160661833073708, 0.058151801834449],
    [0.058151801834449, 0.927437098385088]
]

loglikelihoods = [-1.098653352229586e+03, -1.706951862352565e+03, -1.292882804841197e+03]


def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood
    #####Insert your code here for subtask 6a#####
    N = X.shape[0]
    D = X.shape[1]
    K = covariances.shape[2]
    loglikelihood = 0
    means_array = np.array(means)
    for i in range(N):
        for j in range(K):
            m_diff = X[i] - means_array
            m_diff_tr = np.transpose(m_diff)
            cov = covariances[:, :, j]

            norm_fac = 1 / (((2 * np.pi) ** (D/2)) * (np.linalg.det(cov)) ** 0.5)
            loglikelihood += weights[j] * (norm_fac * (np.exp(-0.5 * m_diff_tr * cov * m_diff)))

    return print(loglikelihood)


for idx in range(1):
    ll = getLogLikelihood(means, weights, covariances, data[idx])

# for idx in range(3):
#     ll = getLogLikelihood(means, weights, covariances, data[idx])
#     diff = loglikelihoods[idx] - ll
#     print('LogLikelihood is {0}, should be {1}, difference: {2}\n'.format(ll, loglikelihoods[idx], diff))
