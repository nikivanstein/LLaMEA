import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 22  # Slightly increased population size for more diversity
        self.mutation_factor = 0.85 + 0.15 * np.random.rand()  # Slightly narrowed mutation factor range
        self.crossover_rate = 0.88  # Adjusted crossover rate for exploration-exploitation balance
        self.initial_temperature = 2.5  # Higher initial temperature for broader initial search
        self.cooling_schedule = lambda t, i: t * (0.83 + 0.12 * np.sin(i * np.pi / 10))  # Modified stochastic cooling schedule
        self.cooling_rate = 0.9  # Lower cooling rate for gradual cooling
        self.chaos_factor = 0.3  # Adjusted chaos influence for maintaining diversity

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        temperature = self.initial_temperature

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                if np.random.rand() < 0.25:  # Introduced randomness in mutation factor adjustment
                    self.mutation_factor = 0.65 + 0.25 * np.random.rand() + self.chaos_factor * np.sin(evaluations / 15)

                if evaluations % 75 == 0:  # Changed frequency for adaptive mutation condition
                    best_idx = np.argmin(fitness)
                    x_best = population[best_idx]
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant_vector = np.clip(x_best + self.mutation_factor * (x2 - x3), self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant_vector = np.clip(x1 + self.mutation_factor * (x2 - x3), self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])

                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                    temperature = self.cooling_schedule(temperature, i)
                else:
                    prob_accept = np.exp((fitness[i] - trial_fitness) / temperature)
                    if np.random.rand() < prob_accept:
                        population[i] = trial_vector
                        fitness[i] = trial_fitness

            temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]