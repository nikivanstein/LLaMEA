import numpy as np

class EnhancedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.alpha = 0.2
        self.beta0 = 1.0
        self.gamma = 0.1
        self.lb = -5.0 * np.ones(dim)
        self.ub = 5.0 * np.ones(dim)
    
    def __call__(self, func):
        def levy_flight():
            beta = self.beta0 / np.sqrt(self.dim)
            sigma1 = (np.prod(np.power(np.arange(1, self.dim+1), -beta)))**(1/self.dim)
            sigma2 = np.power(np.random.standard_normal(self.dim) * sigma1, 1/beta)
            return sigma2
        
        def attraction(x, y):
            r = np.linalg.norm(x - y)
            return np.exp(-self.gamma * r**2)
        
        def levy_update(x):
            step = levy_flight()
            new_x = x + step * np.random.normal(0, 1, self.dim)
            return np.clip(new_x, self.lb, self.ub)
        
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(indiv) for indiv in pop])
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        step_size = attraction(pop[i], pop[j])
                        pop[i] = levy_update(pop[i]) if step_size > np.random.rand() else pop[i]
                        fitness[i] = func(pop[i])
        
        return pop[np.argmin(fitness)]