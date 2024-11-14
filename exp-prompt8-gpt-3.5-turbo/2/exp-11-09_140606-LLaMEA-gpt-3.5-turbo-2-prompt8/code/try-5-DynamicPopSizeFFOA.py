import numpy as np

class DynamicPopSizeFFOA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population_size = 10
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness_values = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget - population_size):
            mean_individual = np.mean(population, axis=0)
            new_individual = mean_individual + np.random.uniform(-1, 1, self.dim) * np.std(population, axis=0)
            new_fitness = func(new_individual)
            
            if new_fitness < np.max(fitness_values):
                max_idx = np.argmax(fitness_values)
                population[max_idx] = new_individual
                fitness_values[max_idx] = new_fitness
            else:
                diversity = np.mean(np.std(population, axis=0))
                if np.random.rand() < 0.5 and population_size < 20:
                    population = np.vstack([population, new_individual])
                    fitness_values = np.append(fitness_values, new_fitness)
                    population_size += 1
                elif np.random.rand() < 0.5 and population_size > 5:
                    min_idx = np.argmin(fitness_values)
                    population = np.delete(population, min_idx, axis=0)
                    fitness_values = np.delete(fitness_values, min_idx)
                    population_size -= 1
        
        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]
        best_fitness = fitness_values[best_idx]
        
        return best_solution, best_fitness