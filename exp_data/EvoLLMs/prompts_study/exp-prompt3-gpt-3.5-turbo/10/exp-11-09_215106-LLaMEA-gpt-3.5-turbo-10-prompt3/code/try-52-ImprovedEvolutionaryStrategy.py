import numpy as np

class ImprovedEvolutionaryStrategy(EvolutionaryStrategy):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 0.01  # Learning rate for adaptive parameters

    def __call__(self, func):
        mu = self.mu
        lambda_ = self.lambda_
        sigma = self.sigma
        pc = self.pc
        ps = self.ps
        C = self.C
        D = self.D
        invsqrtC = self.invsqrtC
        alpha = self.alpha
        
        x_mean = np.random.uniform(-5.0, 5.0, self.dim)
        x = np.random.uniform(-5.0, 5.0, (mu, self.dim))
        fitness = np.array([func(x_i) for x_i in x])
        
        for _ in range(self.budget // lambda_):
            x_old = x.copy()
            fitness_old = fitness.copy()
            
            for i in range(lambda_):
                z = np.random.normal(0, 1, self.dim)
                x[i] = x_mean + sigma * (D * (invsqrtC @ z))
                fitness[i] = func(x[i])
            
            idx = np.argsort(fitness)
            x = x[idx[:mu]]
            x_mean = np.mean(x, axis=0)
            
            z = np.random.normal(0, 1, self.dim)
            ps = (1 - alpha) * ps + np.sqrt(alpha * (2 - alpha)) * (z < 0)
            pc = (1 - 0.4) * pc + np.sqrt(0.4 * (2 - 0.4)) * (z >= 0)
            
            cSigma = (2 * pc * np.sqrt(1 - (1 - pc)**2)) / np.sqrt(self.dim)
            C = np.dot(C, np.dot(np.diagflat(1 - cSigma), C)) + np.outer(cSigma, cSigma)
            D = D * np.exp(0.0873 * (np.linalg.norm(ps) / np.sqrt(self.dim)) - 1)
            invsqrtC = np.linalg.inv(np.linalg.cholesky(C).T)
            
            sigma = sigma * np.exp((np.linalg.norm(ps) - 0.2) / 0.3)
            
        return x[0]