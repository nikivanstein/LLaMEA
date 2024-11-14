import numpy as np

class EnhancedExplorationMutationOptimizationAlgorithmSpeed:
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
                mutation_rate = np.clip(0.5 + np.random.normal(0, 0.1), 0.1, 0.9)
                fitness_range = fitness.max() - fitness.min()
                mutation_rate *= 1 + 0.1 * (fitness_range) / (fitness.max() - fitness.min())
                population[:, i] = 0.8*global_best[i] + 0.2*local_best[i] + mutation_rate * np.random.standard_normal(self.population_size)
                
                if np.random.rand() < 0.3:  # Introducing dynamic mutation strategy
                    diff_vector = np.mean(population, axis=0) - population[i]
                    scale_factor = 0.8 + 0.2 * np.exp(-0.1 * (fitness_range))
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