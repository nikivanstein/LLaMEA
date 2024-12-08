import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.eval_count = 0

    def __call__(self, func):
        self.evaluate_population(func)
        best_index = np.argmin(self.fitness)
        best_solution = self.population[best_index].copy()
        best_fitness = self.fitness[best_index]

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break

                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.population[indices]
                mutant_vector = x0 + self.mutation_factor * (x1 - x2)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Crossover
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, self.population[i])

                # Selection
                trial_fitness = func(trial_vector)
                self.eval_count += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial_vector.copy()

        return best_solution

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.eval_count += 1