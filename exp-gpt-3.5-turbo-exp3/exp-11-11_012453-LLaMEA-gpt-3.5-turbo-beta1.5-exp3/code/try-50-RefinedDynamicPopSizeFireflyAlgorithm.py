import numpy as np

class RefinedDynamicPopSizeFireflyAlgorithm(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.beta0 = 1.5  # Initial beta value for Levy flights
        self.gamma = 1.0  # Initial gamma value for attraction
        self.adaptive_prob = 0.1  # Probability of adaptive population size change
        
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
            
            # Dynamic population size adaptation
            if np.random.rand() < self.adaptive_prob: 
                self.pop_size = min(30, self.pop_size + 5)
                pop = np.vstack((pop, np.random.uniform(self.lb, self.ub, (5, self.dim)))
                fitness = np.append(fitness, [func(indiv) for indiv in pop[-5:]])
        
        return pop[np.argmin(fitness)]