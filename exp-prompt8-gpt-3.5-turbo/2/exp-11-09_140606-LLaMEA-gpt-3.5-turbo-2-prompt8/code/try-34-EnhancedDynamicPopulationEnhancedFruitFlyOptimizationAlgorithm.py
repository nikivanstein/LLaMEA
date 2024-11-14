import numpy as np

class EnhancedDynamicPopulationEnhancedFruitFlyOptimizationAlgorithm:
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
        mutation_strength = np.ones(self.dim)  # Initialize mutation strength per dimension
        
        for _ in range(self.budget - population_size):
            mean_individual = np.mean(population, axis=0)
            mutation = np.random.normal(0, mutation_strength)  # Adaptive mutation based on strength per dimension
            new_individual = mean_individual + mutation
            new_individual = np.clip(new_individual, self.lb, self.ub)  # Ensure within bounds
            new_fitness = func(new_individual)
            
            if new_fitness < np.max(fitness_values):
                max_idx = np.argmax(fitness_values)
                population[max_idx] = new_individual
                fitness_values[max_idx] = new_fitness
                mutation_strength *= 1.1  # Increase mutation strength for better exploration
            else:
                mutation_strength *= 0.9  # Decrease mutation strength for better exploitation
                
            if np.random.rand() < 0.1:
                if new_fitness < np.min(fitness_values):
                    population = np.vstack((population, new_individual))
                    fitness_values = np.append(fitness_values, new_fitness)
                    mutation_strength = np.append(mutation_strength, np.ones(self.dim))  # Initialize new mutation strength
                    population_size += 1
                elif new_fitness < np.max(fitness_values):
                    replace_idx = np.argmax(fitness_values)
                    population[replace_idx] = new_individual
                    fitness_values[replace_idx] = new_fitness

        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]
        best_fitness = fitness_values[best_idx]
        
        return best_solution, best_fitness