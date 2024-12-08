import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        # Initialize population
        population = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations = self.population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_value = fitness[best_index]

        while self.evaluations < self.budget:
            # Adaptively resize population based on remaining budget
            if self.evaluations > self.budget * 0.8:
                self.population_size = max(4, int(self.population_size * 0.8))

            next_population = np.copy(population)

            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)

                # Differential mutation
                a, b, c = population[indices]
                mutant_vector = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                trial_vector = np.copy(population[i])
                crossover_points = np.random.rand(self.dim) < self.CR
                trial_vector[crossover_points] = mutant_vector[crossover_points]

                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                self.evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    next_population[i] = trial_vector
                    fitness[i] = trial_fitness

                    # Update best solution
                    if trial_fitness < best_value:
                        best_solution = trial_vector
                        best_value = trial_fitness

                # Early stopping if budget is exhausted
                if self.evaluations >= self.budget:
                    break

            population = next_population

        return best_solution, best_value