import numpy as np

class EnhancedAdaptiveLearningRateOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        learning_rates = np.full(self.population_size, 0.5)

        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            best_individual = population[sorted_indices[0]]
            global_best = population[sorted_indices[0]]
            local_best = population[sorted_indices[1]]

            for i in range(self.population_size):
                mutation_step = np.random.standard_normal(self.dim) * learning_rates[i]
                trial_vector = population[i] + mutation_step
                trial_fitness = func(trial_vector)

                if trial_fitness < fitness[i]:  # Update individual and learning rate
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                    learning_rates[i] *= 1.1
                else:  # Adjust learning rate downwards
                    learning_rates[i] *= 0.9

            if np.random.rand() < 0.2:  # 20% probability
                new_population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
                new_fitness = np.array([func(individual) for individual in new_population])

                if new_fitness.min() < fitness.min():
                    population = new_population
                    fitness = new_fitness

            # Dynamic population size adjustment based on individual performance
            if np.random.rand() < 0.2: 
                if np.random.rand() < 0.5:
                    self.population_size += 1
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (1, self.dim)))
                    fitness = np.append(fitness, func(population[-1]))
                else:
                    worst_idx = np.argmax(fitness)
                    population = np.delete(population, worst_idx, axis=0)
                    fitness = np.delete(fitness, worst_idx)

        return best_individual