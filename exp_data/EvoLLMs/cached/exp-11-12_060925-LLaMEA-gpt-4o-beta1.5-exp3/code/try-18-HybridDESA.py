import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 1.0
        self.cooling_rate = 0.99

    def __call__(self, func):
        # Initialize population
        population = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations = self.population_size

        best_index = np.argmin(fitness)
        best_value = fitness[best_index]
        best_position = population[best_index]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation and crossover for new candidate
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Fitness evaluation
                trial_fitness = func(trial)
                self.evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best
                    if trial_fitness < best_value:
                        best_value = trial_fitness
                        best_position = trial

                # Simulated annealing adaptation
                if np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if self.evaluations >= self.budget:
                    break

            # Cooling the temperature
            self.temperature *= self.cooling_rate

        return best_position, best_value