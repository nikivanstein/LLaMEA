import numpy as np

class DynamicMutationRateOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            best_individual = population[sorted_indices[0]]

            global_best = population[sorted_indices[0]]
            local_best = population[sorted_indices[1]]

            for i in range(self.dim):
                fitness_diff = np.abs(fitness - fitness[i])
                mutation_rate = 0.5 + np.random.normal(0, 0.1) * (1 + np.sum(fitness_diff)) / (self.population_size * np.sum(fitness_diff))
                population[:, i] = 0.8 * global_best[i] + 0.2 * local_best[i] + mutation_rate * np.random.standard_normal(self.population_size)
                
                if np.random.rand() < 0.3:  # Introducing dynamic mutation rate
                    diff_vector = np.mean(population, axis=0) - population[i]
                    scale_factor = 0.8 + 0.2 * np.exp(-0.1 * np.sum(fitness_diff))
                    trial_vector = population[i] + scale_factor * diff_vector
                    trial_fitness = func(trial_vector)
                    if trial_fitness < fitness[i]:
                        population[i] = trial_vector
                        fitness[i] = trial_fitness
            
            fitness = np.array([func(individual) for individual in population])

            if np.random.rand() < 0.2:  # 20% probability
                new_population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
                new_fitness = np.array([func(individual) for individual in new_population])

                if new_fitness.min() < fitness.min():
                    population = new_population
                    fitness = new_fitness

        return best_individual