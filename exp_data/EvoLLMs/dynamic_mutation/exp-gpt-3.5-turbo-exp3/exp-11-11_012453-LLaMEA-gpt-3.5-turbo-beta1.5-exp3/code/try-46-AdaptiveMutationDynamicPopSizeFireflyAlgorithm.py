import numpy as np

class AdaptiveMutationDynamicPopSizeFireflyAlgorithm(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = 0.2  # Initial mutation rate
        
    def __call__(self, func):
        def adaptive_mutation(fitness):
            return 1 / (1 + np.exp(-self.mutation_rate * (fitness - np.mean(fitness))))
        
        def levy_update(x, fitness):
            step = levy_flight()
            new_x = x + step * np.random.normal(0, 1, self.dim)
            return np.clip(new_x, self.lb, self.ub), adaptive_mutation(fitness)
        
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(indiv) for indiv in pop])
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        step_size = attraction(pop[i], pop[j])
                        pop[i], mutation_rate_i = levy_update(pop[i], fitness[i])
                        fitness[i] = func(pop[i])
                        self.mutation_rate = 0.1 * mutation_rate_i + 0.9 * self.mutation_rate
            
            if np.random.rand() < 0.1:  # Probability of change
                self.pop_size = min(30, self.pop_size + 5)
                pop = np.vstack((pop, np.random.uniform(self.lb, self.ub, (5, self.dim)))
                fitness = np.append(fitness, [func(indiv) for indiv in pop[-5:]])
        
        return pop[np.argmin(fitness)]