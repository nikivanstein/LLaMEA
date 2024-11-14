import numpy as np

class DynamicMutationOptimizationAlgorithm:
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
            
            diversity = np.mean(np.std(population, axis=0))

            for i in range(self.dim):
                mutation_rate = np.clip(0.5 + np.random.normal(0, 0.1), 0.1, 0.9)
                mutation_rate *= 1 + 0.1 * diversity
                population[:, i] = best_individual[i] + mutation_rate * np.random.standard_normal(self.population_size)
                
                if np.random.rand() < 0.3:  # Dynamic mutation based on diversity
                    diff_vector = np.mean(population, axis=0) - population[i]
                    scale_factor = 0.8 + 0.2 * np.exp(-0.1 * diversity)
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