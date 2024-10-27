import numpy as np

class MultipleTrajectorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_trajectories = 5
        self.diversification_rate = 0.2

    def __call__(self, func):
        population = np.random.uniform(low=-5.0, high=5.0, size=(self.num_trajectories, self.dim))
        fitness = np.array([func(individual) for individual in population])

        for _ in range(self.budget - self.num_trajectories):
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            
            for i in range(self.num_trajectories):
                if i != best_idx and np.random.rand() < self.diversification_rate:
                    new_individual = best_individual + np.random.normal(0, 0.1, size=self.dim)
                    new_fitness = func(new_individual)
                    
                    if new_fitness < fitness[i]:
                        population[i] = new_individual
                        fitness[i] = new_fitness
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        return best_solution, best_fitness