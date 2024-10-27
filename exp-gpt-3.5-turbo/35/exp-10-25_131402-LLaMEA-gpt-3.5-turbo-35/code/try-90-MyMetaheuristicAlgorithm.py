import numpy as np

class MyMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.scale_factor = 0.8
        self.prob_refinement = 0.35

    def __call__(self, func):
        pop_size = 10 * self.dim
        lower_bound = -5.0 * np.ones(self.dim)
        upper_bound = 5.0 * np.ones(self.dim)
        
        population = np.random.uniform(low=lower_bound, high=upper_bound, size=(pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget - pop_size):
            for i in range(pop_size):
                indices = np.arange(pop_size)
                indices = np.delete(indices, i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutant = population[a] + self.scale_factor * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.prob_refinement
                trial = np.where(crossover, mutant, population[i])
                
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        return best_solution, best_fitness