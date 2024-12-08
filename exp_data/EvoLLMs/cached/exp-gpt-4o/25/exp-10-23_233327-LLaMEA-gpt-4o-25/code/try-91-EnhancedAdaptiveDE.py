import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_pop_size = 20
        self.mutation_factor = 0.7 + 0.3 * np.random.rand()  # Self-adaptive mutation factor
        self.crossover_rate = 0.85  # Balanced crossover rate for exploration and exploitation
        self.initial_temperature = 2.5  # Start with a higher temperature for wider exploration
        self.cooling_schedule = lambda t, i: t * (0.9 + 0.05 * np.cos(i / 3))  # Smooth cooling schedule
        self.cooling_rate = 0.95  # Gradual cooling for sustained exploration
        self.chaos_factor = 0.3  # Chaos influence for enhancing diversity
        self.dynamic_pop_adjustment = lambda e: self.initial_pop_size + int(10 * np.sin(e / 50))  # Adaptive population size

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.initial_pop_size
        temperature = self.initial_temperature

        while evaluations < self.budget:
            current_pop_size = self.dynamic_pop_adjustment(evaluations)
            for i in range(current_pop_size):
                if evaluations >= self.budget:
                    break

                if evaluations % 50 == 0:
                    self.mutation_factor = 0.5 + 0.4 * np.random.rand() + self.chaos_factor * np.sin(evaluations / 15)

                indices = np.random.choice(self.initial_pop_size, 3, replace=False)
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