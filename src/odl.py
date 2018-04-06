import numpy as np
from sklearn import linear_model
from tqdm import tqdm


class OnlineDictionaryLearning:
    def __init__(self, data, log_step = 40, test_batch_size = 400):
        self.data = data

        self.n_obs = len(self.data)
        self.dim_obs = len(self.data[0])

        self.log_step = log_step
        self.test_batch_size = test_batch_size

        self.losses = []
        self.regret = []
        self.offline_loss = []
        self.objective = []

        self.alphas = []
        self.observed = []
        self.cumulative_losses = []

    def sample(self, data):
        while True :
            permutation = list(np.random.permutation(self.n_obs))
            for idx in permutation:
                yield data[idx]

    def initialize_logs(self):
        self.losses = []
        self.regret = []
        self.offline_loss = []

    @staticmethod
    def compute_alpha(x, dic, lam):
        reg = linear_model.LassoLars(alpha=lam)
        reg.fit(X=dic, y=x)
        return reg.coef_

    @staticmethod
    def compute_dic(a, b, d, dict_size):

        # We run only one iteration for the optimization over D
        # converged = False

        # while not converged :
        for j in range(dict_size) :
            u_j = (b[:, j] - np.matmul(d, a[:, j])) / a[j, j] + d[:, j]
            d[:, j] = u_j / max([1, np.linalg.norm(u_j)])

        return d

    def learn(self, it, lam, dict_size):
        self.initialize_logs()

        data_gen = self.sample(self.data)

        a_prev = 0.01 * np.identity(dict_size)
        b_prev = 0
        d_prev = self.initialize_dic(dict_size, data_gen)

        for it_curr in tqdm(range(it)):
            x = next(data_gen)

            alpha = self.compute_alpha(x, d_prev, lam)

            a_curr = a_prev + np.outer(alpha, alpha.T)
            b_curr = b_prev + np.outer(x, alpha.T)

            d_curr = self.compute_dic(a=a_curr, b=b_curr, d=d_prev, dict_size=dict_size)

            a_prev = a_curr
            b_prev = b_curr
            d_prev = d_curr

            self.log(observation=x, dictionary=d_curr, it=it_curr, lam=lam, alpha=alpha)

        self.compute_objective()

        return d_curr.T

    def log(self, observation, dictionary, it, lam, alpha):
        if it % self.log_step == 0:
            loss = self.one_loss(observation, dictionary, alpha)
            self.losses.append(loss)
            # self.offline_loss.append(self.full_dataset_loss(dictionary, lam))
            self.alphas.append(alpha)
            self.observed.append(observation)
            self.cumulative_losses.append(self.cumulative_loss(dictionary))

    def cumulative_loss(self, dictionary):
        n_observed = len(self.observed)
        return sum([self.one_loss(self.observed[i], dictionary, self.alphas[i])
                    for i in range(n_observed)]) / n_observed

    @staticmethod
    def one_loss(x, dictionary, alpha):
        return np.linalg.norm(x - np.matmul(dictionary, alpha), ord=2) ** 2

    @staticmethod
    def initialize_dic(dict_size, data_gen):
        return np.array([next(data_gen) for _ in range(dict_size)]).T

    def observation_loss(self, x, dictionary, lam):
        alpha = self.compute_alpha(x, dictionary, lam)
        return np.linalg.norm(x - np.matmul(dictionary, alpha), ord=2) ** 2

    def full_dataset_loss(self, dictionary, lam):
        data_gen = self.sample(self.data)
        return sum([self.observation_loss(next(data_gen), dictionary, lam)
                    for _ in range(self.test_batch_size)])

    def compute_objective(self):
        cumulated_loss = np.cumsum(self.losses)
        self.objective = [cumulated_loss[i] / (i+1)
                          for i in range(len(cumulated_loss))]