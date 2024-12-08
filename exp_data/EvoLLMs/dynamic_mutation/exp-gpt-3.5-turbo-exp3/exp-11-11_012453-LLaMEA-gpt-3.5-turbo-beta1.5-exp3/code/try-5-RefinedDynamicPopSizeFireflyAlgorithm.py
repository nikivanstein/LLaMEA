import numpy as np

class RefinedDynamicPopSizeFireflyAlgorithm(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 0.1  # Step size for Levy flight
        
    def levy_flight(self):
        beta = self.beta0 / np.sqrt(self.dim)
        sigma1 = (np.prod(np.power(np.arange(1, self.dim+1), -beta)))**(1/self.dim)
        sigma2 = np.power(np.random.standard_normal(self.dim) * sigma1, 1/beta)
        return self.alpha * sigma2
    
    def __call__(self, func):
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(indiv) for indiv in pop])
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        step_size = self.attraction(pop[i], pop[j])
                        pop[i] = self.levy_update(pop[i]) if step_size > np.random.rand() else pop[i]
                        fitness[i] = func(pop[i])
            
            # Dynamic population size adaptation
            if np.random.rand() < 0.1:  # Probability of change
                self.pop_size = min(30, self.pop_size + 5)
                pop = np.vstack((pop, np.random.uniform(self.lb, self.ub, (5, self.dim)))
                fitness = np.append(fitness, [func(indiv) for indiv in pop[-5:]])
        
        return pop[np.argmin(fitness)]