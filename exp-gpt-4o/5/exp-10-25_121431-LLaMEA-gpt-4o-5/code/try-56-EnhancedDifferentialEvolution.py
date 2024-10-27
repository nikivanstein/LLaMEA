import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 8 + int(self.dim * np.log(self.dim)))  # Slightly adjusted population size
        self.mutation_factor = 0.6 + np.random.rand(self.population_size) * 0.2  # Tightened mutation factor range
        self.crossover_rate = 0.65 + np.random.rand(self.population_size) * 0.3  # Adjusted range for crossover rate
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
            current_population_size = min(self.population_size, self.budget - self.eval_count)
            
            for i in range(current_population_size):
                if self.eval_count >= self.budget:
                    break

                indices = np.random.choice(current_population_size, 3, replace=False)
                x0, x1, x2 = self.population[indices]
                if np.random.rand() < 0.05:  # Probability for gradient boosting
                    gradient = np.random.randn(self.dim) * 0.2 * (x0 - best_solution)  # Increased gradient influence
                else:
                    gradient = np.zeros(self.dim)
                mutant_vector = x0 + self.mutation_factor[i] * (x1 - x2) + gradient
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate[i], mutant_vector, self.population[i])

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