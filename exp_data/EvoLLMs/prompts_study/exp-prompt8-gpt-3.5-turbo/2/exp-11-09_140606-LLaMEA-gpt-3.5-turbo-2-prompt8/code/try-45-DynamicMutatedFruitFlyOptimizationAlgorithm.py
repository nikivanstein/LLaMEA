import numpy as np

class DynamicMutatedFruitFlyOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.step_size = 1.0

    def __call__(self, func):
        population_size = 10
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness_values = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget - population_size):
            mean_individual = np.mean(population, axis=0)
            new_individual = mean_individual + np.random.uniform(-1, 1, self.dim) * self.step_size
            new_fitness = func(new_individual)
            
            diff = np.abs(new_individual - mean_individual)
            mutation_step = np.clip(np.mean(diff), 0.1, 1.0)
            new_individual = mean_individual + np.random.uniform(-1, 1, self.dim) * mutation_step
            new_fitness = func(new_individual)
            
            if new_fitness < np.max(fitness_values):
                max_idx = np.argmax(fitness_values)
                population[max_idx] = new_individual
                fitness_values[max_idx] = new_fitness
            
            if np.random.rand() < 0.1:
                replace_idx = np.argmax(fitness_values)
                population[replace_idx] = new_individual
                fitness_values[replace_idx] = new_fitness

        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]
        best_fitness = fitness_values[best_idx]
        
        return best_solution, best_fitness