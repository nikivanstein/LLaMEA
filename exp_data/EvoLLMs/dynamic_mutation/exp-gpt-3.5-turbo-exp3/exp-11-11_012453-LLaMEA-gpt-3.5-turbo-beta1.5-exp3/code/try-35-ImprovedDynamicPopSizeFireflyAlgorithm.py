import numpy as np

class ImprovedDynamicPopSizeFireflyAlgorithm(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.beta0 = 1.5  # Initial beta parameter for Levy flight
        
    def levy_flight(self, beta):
        beta = beta / np.sqrt(self.dim)
        sigma1 = (np.prod(np.power(np.arange(1, self.dim+1), -beta)))**(1/self.dim)
        sigma2 = np.power(np.random.standard_normal(self.dim) * sigma1, 1/beta)
        return sigma2
        
    def __call__(self, func):
        beta = self.beta0  # Initialize beta parameter
        for _ in range(self.budget):
            step_sizes = [self.levy_flight(beta) for _ in range(self.pop_size)]
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        step_size = attraction(pop[i], pop[j])
                        pop[i] = levy_update(pop[i], step_sizes[i]) if step_size > np.random.rand() else pop[i]
                        fitness[i] = func(pop[i])
            if np.random.rand() < 0.1:  # Probability of change
                self.pop_size = min(30, self.pop_size + 5)
                beta = max(0.5, beta * 0.95)  # Adapt beta parameter
                pop = np.vstack((pop, np.random.uniform(self.lb, self.ub, (5, self.dim)))
                fitness = np.append(fitness, [func(indiv) for indiv in pop[-5:]])
        return pop[np.argmin(fitness)]