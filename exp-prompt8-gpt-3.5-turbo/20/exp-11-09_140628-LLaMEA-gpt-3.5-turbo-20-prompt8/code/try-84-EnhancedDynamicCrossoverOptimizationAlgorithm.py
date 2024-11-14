import numpy as np

class EnhancedDynamicCrossoverOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.min_population_size = 5
        self.max_population_size = 15

    def __call__(self, func):
        population_size = self.min_population_size
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        crossover_probs = np.full(population_size, 0.9)  # Initialize crossover probabilities

        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            best_individual = population[sorted_indices[0]]
            global_best = population[sorted_indices[0]]
            local_best = population[sorted_indices[1]]

            for i in range(population_size):
                if np.random.rand() < crossover_probs[i]:  # Dynamic probability for crossover
                    mutation_step = np.random.standard_normal(self.dim)
                    trial_vector = population[i] + mutation_step
                    trial_fitness = func(trial_vector)

                    if trial_fitness < fitness[i]:  # Update individual
                        population[i] = trial_vector
                        fitness[i] = trial_fitness
                        crossover_probs[i] *= 1.1  # Adjust crossover probability based on success
                    else:
                        crossover_probs[i] *= 0.9

            if np.random.rand() < 0.2:  # 20% probability for new population generation
                new_population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
                new_fitness = np.array([func(individual) for individual in new_population])

                if new_fitness.min() < fitness.min():
                    population = new_population
                    fitness = new_fitness

            if np.random.rand() < 0.15:  # 15% probability for dynamic step size adjustment
                for i in range(population_size):
                    if fitness[i] < np.mean(fitness):
                        crossover_probs[i] *= 1.2
                    else:
                        crossover_probs[i] *= 0.8

            if np.random.rand() < 0.1:  # 10% probability for dynamic population size adjustment
                if np.random.rand() < 0.5 and population_size < self.max_population_size:
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (1, self.dim))))
                    fitness = np.append(fitness, func(population[-1]))
                    crossover_probs = np.append(crossover_probs, 0.9)
                    population_size += 1
                elif population_size > self.min_population_size:
                    worst_idx = np.argmax(fitness)
                    population = np.delete(population, worst_idx, axis=0)
                    fitness = np.delete(fitness, worst_idx)
                    crossover_probs = np.delete(crossover_probs, worst_idx)
                    population_size -= 1

        return best_individual