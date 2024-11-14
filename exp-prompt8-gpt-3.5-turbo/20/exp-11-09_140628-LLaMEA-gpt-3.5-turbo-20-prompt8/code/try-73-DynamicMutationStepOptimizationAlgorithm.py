import numpy as np

class DynamicMutationStepOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        mutation_step_size = np.full(self.population_size, 0.5)

        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            best_individual = population[sorted_indices[0]]
            global_best = population[sorted_indices[0]]
            local_best = population[sorted_indices[1]]

            for i in range(self.population_size):
                mutation_step = np.random.standard_normal(self.dim) * mutation_step_size[i]
                trial_vector = population[i] + mutation_step
                trial_fitness = func(trial_vector)

                if trial_fitness < fitness[i]:  # Update individual and mutation step size
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                    mutation_step_size[i] *= 1.2
                else:  # Adjust mutation step size downwards
                    mutation_step_size[i] *= 0.8

            if np.random.rand() < 0.2:  # 20% probability
                new_population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
                new_fitness = np.array([func(individual) for individual in new_population])

                if new_fitness.min() < fitness.min():
                    population = new_population
                    fitness = new_fitness

        return best_individual