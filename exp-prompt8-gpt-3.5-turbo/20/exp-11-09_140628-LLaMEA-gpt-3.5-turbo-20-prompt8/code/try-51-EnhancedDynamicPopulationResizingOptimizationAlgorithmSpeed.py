import numpy as np

class EnhancedDynamicPopulationResizingOptimizationAlgorithmSpeed:
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
                fitness_diff = (best_individual - population) @ (best_individual - population).T
                mutation_rate *= 1 + 0.1 * (fitness - fitness.min()) / (fitness.max() - fitness.min())
                population[:, i] = 0.8*global_best[i] + 0.2*local_best[i] + mutation_rate * np.random.standard_normal(self.population_size)

                if np.random.rand() < 0.3:  # Introducing a dynamic mutation strategy
                    local_fitness = np.array([func(population[j]) for j in range(self.population_size)])
                    local_best_index = np.argmin(local_fitness)
                    diff_vector = population[local_best_index] - population[i]
                    trial_vector = population[i] + 0.8*diff_vector
                    trial_fitness = func(trial_vector)
                    if trial_fitness < fitness[i]:
                        population[i] = trial_vector
                        fitness[i] = trial_fitness

            # Introducing adaptive search space transformation
            for i in range(self.dim):
                transformed_population = np.clip(2 * population - best_individual, -5.0, 5.0)
                transformed_fitness = np.array([func(individual) for individual in transformed_population])
                if transformed_fitness.min() < fitness.min():
                    population = transformed_population
                    fitness = transformed_fitness

        return best_individual