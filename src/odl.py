import numpy as np
from sklearn import linear_model
from tqdm import tqdm


class OnlineDictionaryLearning:
    def __init__(self, data):
        self.data = data

        self.n_obs = len(self.data)
        self.dim_obs = len(self.data[0])

    def sample(self, data):
        while True :
            permutation = list(np.random.permutation(self.n_obs))
            for idx in permutation:
                yield data[idx]

    @staticmethod
    def compute_alpha(x, dic, lam):
        reg = linear_model.LassoLars(alpha=lam)
        reg.fit(X=dic, y=x)
        return reg.coef_

    @staticmethod
    def compute_dic(a, b, d, dict_size):

        converged = False

        while not converged :
            for j in range(dict_size) :
                u_j = (b[:, j] - np.matmul(d, a[:, j])) / a[j, j] + d[:, j]
                d[:, j] = u_j / max([1, np.linalg.norm(u_j)])

        return d

    def learn(self, it, lam, dict_size):
        data_gen = self.sample(self.data)

        a_prev = 0
        b_prev = 0
        d_prev = self.initialize_dic(dict_size)

        for _ in tqdm(range(it)):
            x = next(data_gen)

            alpha = self.compute_alpha(x, d_prev, lam)

            a_curr = a_prev + np.outer(alpha, alpha.T)
            b_curr = b_prev + np.outer(x, alpha.T)

            d_curr = self.compute_dic(a=a_curr, b=b_curr, d=d_prev, dict_size=dict_size)

            a_prev = a_curr
            b_prev = b_curr
            d_prev = d_curr

        return d_curr

    def initialize_dic(self, dict_size):
        return np.random.rand(self.dim_obs, dict_size) * 2 - 1

    def loss(self, alpha, d, lam):
        data_gen = self.sample(self.data)
        return sum([self.loss_obs(x= next(data_gen), alpha=alpha, d=d)
                    for i in range(self.n_obs)]) \
               + lam * np.linalg.norm(alpha, ord=1)

    @staticmethod
    def loss_obs(x, alpha, d):
        return np.linalg.norm(x - np.matmul(d, alpha), ord=2) ** 2