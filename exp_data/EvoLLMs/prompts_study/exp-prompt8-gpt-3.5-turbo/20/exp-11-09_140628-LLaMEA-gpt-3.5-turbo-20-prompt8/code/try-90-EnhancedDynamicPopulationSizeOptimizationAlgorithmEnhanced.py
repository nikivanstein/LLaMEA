import numpy as np

class EnhancedDynamicPopulationSizeOptimizationAlgorithmEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.min_population_size = 5
        self.max_population_size = 15

    def __call__(self, func):
        population_size = self.min_population_size
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        learning_rates = np.full(population_size, 0.5)

        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            best_individual = population[sorted_indices[0]]
            global_best = population[sorted_indices[0]]
            local_best = population[sorted_indices[1]]

            for i in range(population_size):
                if np.random.rand() < 0.5:  # 50% probability for mutation
                    mutation_step = np.random.standard_normal(self.dim) * learning_rates[i]
                    trial_vector = population[i] + mutation_step
                    trial_fitness = func(trial_vector)

                    if trial_fitness < fitness[i]:  # Update individual and learning rate
                        population[i] = trial_vector
                        fitness[i] = trial_fitness
                        learning_rates[i] *= 1.1
                    else:  # Adjust learning rate based on trial performance
                        if trial_fitness < np.mean(fitness):
                            learning_rates[i] *= 1.2
                        else:
                            learning_rates[i] *= 0.8

            if np.random.rand() < 0.2:  # 20% probability
                new_population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
                new_fitness = np.array([func(individual) for individual in new_population])

                if new_fitness.min() < fitness.min():
                    population = new_population
                    fitness = new_fitness

            if np.random.rand() < 0.15:  # 15% probability for dynamic step size adjustment based on fitness improvement
                for i in range(population_size):
                    if fitness[i] < np.mean(fitness):  # Increase step size if individual fitness is below average
                        learning_rates[i] *= 1.2
                    else:
                        learning_rates[i] *= 0.8

            if np.random.rand() < 0.1:  # 10% probability for dynamic population size adjustment
                if np.random.rand() < 0.5 and population_size < self.max_population_size:
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (1, self.dim))))
                    fitness = np.append(fitness, func(population[-1]))
                    learning_rates = np.append(learning_rates, 0.5)
                    population_size += 1
                elif population_size > self.min_population_size:
                    worst_idx = np.argmax(fitness)
                    population = np.delete(population, worst_idx, axis=0)
                    fitness = np.delete(fitness, worst_idx)
                    learning_rates = np.delete(learning_rates, worst_idx)
                    population_size -= 1

            if np.random.rand() < 0.1:  # 10% probability for probabilistic crowding mechanism
                crowding_population = np.copy(population)
                crowding_fitness = np.copy(fitness)
                for j in range(population_size):
                    rand_idx = np.random.choice(np.delete(np.arange(population_size), j))
                    if crowding_fitness[j] < fitness[rand_idx]:
                        population[j] = crowding_population[j]
                        fitness[j] = crowding_fitness[j]

        return best_individual