import numpy as np

class DynamicPopSizeFireflyAlgorithmImproved(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.gamma = 1.0  # Initial gamma value

    def __call__(self, func):
        def update_gamma(fitness_values):
            mean_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)
            if std_fitness > 0:
                self.gamma = 1.0 / (1.0 + np.exp(-0.1 * (mean_fitness - std_fitness)))
        
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(indiv) for indiv in pop])
        
        for _ in range(1, self.budget):
            update_gamma(fitness)
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        step_size = attraction(pop[i], pop[j])
                        pop[i] = levy_update(pop[i]) if step_size > np.random.rand() else pop[i]
                        fitness[i] = func(pop[i])
            
            if np.random.rand() < 0.15:  # Probability of change
                self.pop_size = min(30, self.pop_size + 5)
                pop = np.vstack((pop, np.random.uniform(self.lb, self.ub, (5, self.dim)))
                fitness = np.append(fitness, [func(indiv) for indiv in pop[-5:]])
        
        return pop[np.argmin(fitness)]