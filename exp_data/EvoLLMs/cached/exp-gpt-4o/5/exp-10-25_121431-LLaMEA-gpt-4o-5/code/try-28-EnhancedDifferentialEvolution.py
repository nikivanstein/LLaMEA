import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4, 5 + int(np.sqrt(self.dim * np.log(self.dim))))  # Improved dynamic population size
        self.mutation_factor = np.random.rand(self.population_size) * 0.6 + 0.3  # Adaptive mutation factor with wider range
        self.crossover_rate = np.random.rand(self.population_size) * 0.3 + 0.7  # Adaptive crossover rate with narrower range
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

                # Mutation with dynamic strategy
                strategy_selector = np.random.rand()
                if strategy_selector < 0.5:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    x0, x1, x2 = self.population[indices]
                    perturbation = np.random.randn(self.dim) * 0.05 * (best_solution - x0)
                else:
                    indices = np.random.choice(self.population_size, 5, replace=False)
                    x0, x1, x2, x3, x4 = self.population[indices]
                    perturbation = np.random.randn(self.dim) * 0.05 * (best_solution - x0)
                    x0 = x0 + self.mutation_factor[i] * (x1 - x2) + self.mutation_factor[i] * (x3 - x4)

                mutant_vector = x0 + perturbation
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Crossover with adaptive rate
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate[i], mutant_vector, self.population[i])

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