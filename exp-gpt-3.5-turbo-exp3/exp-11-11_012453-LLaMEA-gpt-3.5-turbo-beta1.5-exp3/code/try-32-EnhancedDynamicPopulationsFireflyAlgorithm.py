import numpy as np

class EnhancedDynamicPopulationsFireflyAlgorithm(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        
    def __call__(self, func):
        def update_pop_size(pop, fitness):
            sorted_indices = np.argsort(fitness)
            best_indices = sorted_indices[:self.pop_size]
            best_pop = pop[best_indices]
            worst_indices = sorted_indices[self.pop_size:]
            worst_pop = pop[worst_indices]
            
            avg_best_fitness = np.mean(fitness[best_indices])
            avg_worst_fitness = np.mean(fitness[worst_indices])
            improvement_ratio = avg_best_fitness / avg_worst_fitness
            
            if improvement_ratio > 1.1:  # Threshold for improvement
                self.pop_size = min(30, self.pop_size + 5)
                pop = best_pop
                fitness = np.array([func(indiv) for indiv in pop])
            
            return pop, fitness
        
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(indiv) for indiv in pop])
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        step_size = attraction(pop[i], pop[j])
                        pop[i] = levy_update(pop[i]) if step_size > np.random.rand() else pop[i]
                        fitness[i] = func(pop[i])
            
            pop, fitness = update_pop_size(pop, fitness)
        
        return pop[np.argmin(fitness)]