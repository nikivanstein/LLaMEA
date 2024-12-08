import numpy as np

class HybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.mutation_factor = 0.75 + 0.3 * np.random.rand()  # Adjusted mutation factor for more adaptability
        self.crossover_rate = 0.85  # Slightly reduced crossover rate for more stability
        self.initial_temperature = 3.0  # Higher initial temperature for broader search
        self.cooling_schedule = lambda t, i: t * (0.9 + 0.1 * np.cos(i / 6))  # Sine-based cooling schedule
        self.cooling_rate = 0.88  # Altered cooling rate
        self.momentum_factor = 0.1  # Introduced momentum influence for dynamic exploration

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        temperature = self.initial_temperature
        momentum = np.zeros((self.pop_size, self.dim))

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                if evaluations % 50 == 0:  # Frequency for dynamic mutation
                    self.mutation_factor = 0.5 + 0.4 * np.random.rand()

                if evaluations % 80 == 0:  # Condition for adaptive mutation
                    best_idx = np.argmin(fitness)
                    x_best = population[best_idx]
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant_vector = np.clip(x_best + self.mutation_factor * (x2 - x3) + momentum[i], self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant_vector = np.clip(x1 + self.mutation_factor * (x2 - x3) + momentum[i], self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    momentum[i] = self.momentum_factor * (trial_vector - population[i])
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                    temperature = self.cooling_schedule(temperature, i)
                else:
                    prob_accept = np.exp((fitness[i] - trial_fitness) / temperature)
                    if np.random.rand() < prob_accept:
                        momentum[i] = self.momentum_factor * (trial_vector - population[i])
                        population[i] = trial_vector
                        fitness[i] = trial_fitness

            temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]