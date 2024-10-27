import numpy as np

class ProbabilisticRefinedEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.prob_refinement = 0.35

    def __call__(self, func):
        pop_size = 10 * self.dim
        lower_bound = -5.0 * np.ones(self.dim)
        upper_bound = 5.0 * np.ones(self.dim)
        
        population = np.random.uniform(low=lower_bound, high=upper_bound, size=(pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget - pop_size):
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            new_individual = best_individual + np.random.normal(0, 0.1, size=self.dim)
            
            if np.random.rand() < self.prob_refinement:
                new_individual = best_individual + np.random.normal(0, 0.05, size=self.dim)  # Refinement step
            
            new_fitness = func(new_individual)
            
            if new_fitness < fitness[best_idx]:
                population[best_idx] = new_individual
                fitness[best_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        return best_solution, best_fitness