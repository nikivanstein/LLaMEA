import numpy as np

class EvolutionaryStrategyImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mu = 10  # population size
        self.lambda_ = 20  # offspring size
        self.sigma = 1.0
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.C = np.eye(dim)
        self.D = np.ones(dim)
        self.invsqrtC = np.linalg.inv(np.linalg.cholesky(self.C).T)
        self.individual_learning_rates = np.ones((self.lambda_, self.dim))  # Individual learning rates
        
    def __call__(self, func):
        mu = self.mu
        lambda_ = self.lambda_
        sigma = self.sigma
        pc = self.pc
        ps = self.ps
        C = self.C
        D = self.D
        invsqrtC = self.invsqrtC
        individual_learning_rates = self.individual_learning_rates
        
        x_mean = np.random.uniform(-5.0, 5.0, self.dim)
        x = np.random.uniform(-5.0, 5.0, (mu, self.dim))
        fitness = np.array([func(x_i) for x_i in x])
        
        for _ in range(self.budget // lambda_):
            x_old = x.copy()
            fitness_old = fitness.copy()
            
            for i in range(lambda_):
                z = np.random.normal(0, 1, self.dim)
                x[i] = x_mean + sigma * (D * (invsqrtC @ z)) * individual_learning_rates[i]  # Dynamic mutation strategy
                fitness[i] = func(x[i])
            
            idx = np.argsort(fitness)
            x = x[idx[:mu]]
            x_mean = np.mean(x, axis=0)
            
            z = np.random.normal(0, 1, self.dim)
            ps = (1 - 0.1) * ps + np.sqrt(0.1 * (2 - 0.1)) * (z < 0)
            pc = (1 - 0.4) * pc + np.sqrt(0.4 * (2 - 0.4)) * (z >= 0)
            
            cSigma = (2 * pc * np.sqrt(1 - (1 - pc)**2)) / np.sqrt(self.dim)
            C = np.dot(C, np.dot(np.diagflat(1 - cSigma), C)) + np.outer(cSigma, cSigma)
            D = D * np.exp(0.0873 * (np.linalg.norm(ps) / np.sqrt(self.dim)) - 1)
            invsqrtC = np.linalg.inv(np.linalg.cholesky(C).T)
            
            sigma = sigma * np.exp((np.linalg.norm(ps) - 0.2) / 0.3)
            
        return x[0]