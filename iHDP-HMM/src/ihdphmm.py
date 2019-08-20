__author__ = 'jinjun sun'
__email__ = 'jsunster@gmail.com'
__doc__ = """This code was derived from sticky hdp-hmm with stick kappa=0. For spatial temporal
             analysis, we use scipy.spatial and rtree(shapely)"""

import numpy as np
from numpy.random import (choice, normal, dirichlet, beta, 
                          gamma, multinomial, exponential, binomial)
from collections import defaultdict
from scipy.cluster.vq import kmeans2
# import ipdb


"""
Dirichlet process (DP)
"""
class DP:
    """
    Initialization of DP
    alpha: initial learning rate
    """
    def __init__(self, alpha, G=None, k=None):
        """
        k is integer, dimesion, k should be defined early alpha
        is integer decist the shape if distribution theata
        """
        self.k = len(G)
        self.alpha = alpha
        self.G = G

    """
    Definition of stick-breaking process
    alpha: initial learning rate
    k: integer dimension of G
    """
    def stick_breaking(self, alpha, k):
        betas = beta(1, alpha, k)
        remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
        p = betas * remaining_pieces
        return p/p.sum()

    """
    Sampling method by using stick-breaking process
    """
    def sample(self):
        p = self.stick_breaking(self.alpha, self.k)
        x = multinomial(self.k - 1, p)
        try:
            return self.G[x]
        except:
            ipdb.set_trace()

"""
Improved Hierarchical Dirichlet Process - Hidden Markov Model (iHDP-HMM).
Data interface should be modify to improve the generalization of the code.
Such as: data from sqlite3 or CSV and other data source
"""
class iHDPHMM:
    """
    Initialization of iHDP-HMM
    """
    def __init__(self, docs, gma=1, rho=1, xi=1, alpha=1, sigma_a=2, sigma_b=2,
                 nu=2, L=20, kappa=0, kmeans_init=True):
        self.docs = docs
        self.L = L # max number for topics
        stickbreaking = self._gem(gma)
        self.w0 = np.array([next(stickbreaking) for i in range(L)])
        self.rho = rho
        self.xi = xi
        self.alpha = alpha        
        self.gma = gma
        # Hyperparameters
        self.nu = nu
        self.a = sigma_a
        self.b = sigma_b


    """
    Compute log-likelihood.
    x: data
    mu: mean
    simage: variance
    """
    def _logphi(self, x, mu, sigma):
        return -((x - mu) / sigma) ** 2 / 2 - np.log(sigma)
        
    """
    Generate the stick-breaking process with parameter gma.
    """
    def _gem(self, gma):
        prev = 1
        while True: 
            # replace True with 1 to faster
            beta_k = beta(1, gma) * prev
            prev -= beta_k
            yield beta_k

    """
    Run blocked-Gibbs sampling is in a loop, so need for a global loop
    """
    def sampler(self):
        self.C = np.random.randint(1, 31)
        tmp = [self.xi/float(self.C)]*self.C
        Q = dirichlet(tmp)
        ## cj in the doc_labels
        self.doc_labels = np.random.choice(range(self.C), size=len(self.docs), p=Q)
        doc_cluster = zip(self.doc_labels, self.docs)
        doc_dict = defaultdict(list)
        for cj, doc in doc_cluster:
            doc_dict[cj].append(doc)

        for cj, ldocs in doc_dict.items():
            self.data = np.concatenate(ldocs)
            self.n = len(self.data)
            self.state = np.random.choice(self.L, self.n)
            std = np.std(self.data)
            self.mu = normal(0, std, self.L)
            self.sigma = np.ones(self.L) * std
            for i in range(self.L):
                idx = np.where(self.state == i)
                if idx[0].size:
                    cluster = self.data[idx]
                    self.mu[i] = np.mean(cluster)
                    self.sigma[i] = np.std(cluster)
            self.N = np.zeros((self.L, self.L))
            for i in range(self.n):
                self.N[self.state[i], self.state[i]] += 1

            self.M = np.zeros(self.N.shape)
            self.PI = (self.N.T / (np.sum(self.N, axis=1) + 1e-7)).T


            dp0 = DP(self.rho, self.w0)
            w_cj = dp0.sample()
            # todo:  put all this into sampling

            for obs in range(self.n):
                # Step 1: messages
                # Notes: L should be global topics, T documents
                messages = np.zeros(self.L)
                messages[-1] = 1
                messages = self.PI.dot(messages * np.exp(self._logphi(self.data[obs], self.mu, self.sigma)))
                messages /= np.max(messages)

                # Step 2: states by MH algorithm
                j = choice(self.L) # proposal
                k = self.state[obs]

                logprob_accept = (np.log(messages[k]) -
                                  np.log(messages[j]) +
                                  np.log(self.PI[self.state[obs], k]) -
                                  np.log(self.PI[self.state[obs], j]) +
                                  self._logphi(self.data[obs],
                                               self.mu[k],
                                               self.sigma[k]) -
                                  self._logphi(self.data[obs],
                                               self.mu[j],
                                               self.sigma[j]))
                if exponential(1) > logprob_accept:
                    self.state[obs] = j
                    self.N[self.state[obs], j] += 1
                    self.N[self.state[obs], k] -= 1

            # Step 3: auxiliary variables
            P = np.tile(w_cj, (self.L, 1)) + self.n
            # np.fill_diagonal(P, np.diag(P) + self.kappa)
            np.fill_diagonal(P, np.diag(P))
            P = 1 - self.n / P
            for i in range(self.L):
                for j in range(self.L):
                    self.M[i, j] = binomial(self.M[i, j], P[i, j])

            w = np.array([binomial(self.M[i, i], 1 / (1 + w_cj[i])) for i in range(self.L)])
            m_bar = np.sum(self.M, axis=0) - w

            # Step 4: beta and parameters of clusters
            # todo: modifiy here to change beta to w_cj
            # self.beta = dirichlet(np.ones(self.L) * (self.gma / self.L + m_bar))
            # self.beta = dp1.sample()
            # Step 5: transition matrix
            # todo: w_cj will replace self.beta here
            dp1 = DP(self.alpha, w_cj)
            wm_cj = dp1.sample()
            self.PI =  np.tile(self.alpha * wm_cj, (self.L, 1)) + self.N
            # np.fill_diagonal(self.PI, np.diag(self.PI) + self.kappa)

            for i in range(self.L):
                self.PI[i, :] = dirichlet(self.PI[i, :])
                idx = np.where(self.state == i)
                cluster = self.data[idx]
                nc = cluster.size
                if nc:
                    xmean = np.mean(cluster)
                    self.mu[i] = xmean / (self.nu / nc + 1)
                    self.sigma[i] = (2 * self.b + (nc - 1) * np.var(cluster) +
                                     nc * xmean ** 2 / (self.nu + nc)) / (2 * self.a + nc - 1)
                else:
                    self.mu[i] = normal(0, np.sqrt(self.nu))
                    self.sigma[i] = 1 / gamma(self.a, self.b)
                        
    """
    Get the estimated sample path of h.
    """
    def getPath(self, h):
        paths = np.zeros(self.data.shape[0])
        for i, mu in enumerate(self.mu):
            paths[np.where(self.state[:, h] == i)] = mu
        return paths
