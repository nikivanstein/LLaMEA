import numpy as np

class GAMBS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.base_pop_size = 10 * dim
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.memory_factor = 0.3

    def __call__(self, func):
        population_size = self.base_pop_size
        population = self.lower_bound + np.random.rand(population_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations = population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        historical_population = population.copy()

        while self.evaluations < self.budget:
            new_population = []
            for _ in range(population_size):
                if self.evaluations >= self.budget:
                    break

                # Tournament selection
                indices = np.random.choice(population_size, 3, replace=False)
                parent1, parent2 = population[indices[:2]], population[indices[1:]]
                
                # Crossover
                if np.random.rand() < self.crossover_rate:
                    crossover_point = np.random.randint(1, self.dim)
                    offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                else:
                    offspring = parent1.copy()

                # Adaptive mutation based on historical data
                if np.random.rand() < self.mutation_rate:
                    mutation_scale = self.memory_factor * np.std(historical_population, axis=0)
                    mutation_vector = np.random.normal(0, mutation_scale, self.dim)
                    offspring += mutation_vector
                    offspring = np.clip(offspring, self.lower_bound, self.upper_bound)

                trial_fitness = func(offspring)
                self.evaluations += 1

                if trial_fitness < best_fitness:
                    best_individual = offspring
                    best_fitness = trial_fitness

                new_population.append(offspring)

            population = np.array(new_population)
            fitness = np.apply_along_axis(func, 1, population)

            # Store the best solutions into historical memory
            historical_population = np.vstack((historical_population, population))
            historical_population = historical_population[np.argsort(np.apply_along_axis(func, 1, historical_population))[:population_size]]

        return best_individual, best_fitness