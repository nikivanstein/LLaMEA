import numpy as np

class FasterConvergingDynamicPopulationSizeAdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_rate = np.full(self.pop_size, 0.1)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = [func(ind) for ind in population]
        
        for _ in range(self.budget):
            offspring = []
            selected_indices = np.random.choice(self.pop_size, size=self.pop_size, replace=True, p=[1 - fit / sum(fitness) for fit in fitness])
            for i in range(self.pop_size):
                parent = population[selected_indices[i]]
                child = parent + self.mutation_rate[i] * np.random.randn(self.dim)
                if func(child) < fitness[selected_indices[i]]:
                    population[selected_indices[i]] = child
                    fitness[selected_indices[i]] = func(child)
                    self.mutation_rate[selected_indices[i]] *= 1.02 + 0.005 * np.mean(self.mutation_rate)  
                    self.pop_size = int(10 * (1 - np.mean(self.mutation_rate)))  
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))))  
                else:
                    self.mutation_rate[selected_indices[i]] *= 0.98 - 0.005 * np.mean(self.mutation_rate)  
                    self.pop_size = int(10 * (1 - np.mean(self.mutation_rate)))  
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))))   
        return population[np.argmin(fitness)]