import numpy as np

class EnhancedChaosInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 25
        self.mutation_factor = 0.7 + 0.3 * np.random.rand()  # Empowered self-adaptive mutation factor
        self.crossover_rate = 0.85  # Slightly reduced crossover rate for precision
        self.initial_temperature = 2.5  # Enhanced initial temperature for expansive search
        self.cooling_schedule = lambda t, i: t * (0.8 + 0.15 * np.cos(i / 7))  # Refined stochastic cooling schedule
        self.cooling_rate = 0.88  # Further adjusted cooling rate for balance
        self.chaos_factor = 0.4  # Increased chaos influence for better diversity

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

                if evaluations % 50 == 0:  # Increased frequency for dynamic mutation adjustments
                    self.mutation_factor = 0.65 + 0.35 * np.random.rand() + self.chaos_factor * np.sin(evaluations / 12)

                if evaluations % 80 == 0:  # Altered condition for best vector-driven adaptive mutation
                    best_idx = np.argmin(fitness)
                    x_best = population[best_idx]
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant_vector = np.clip(x_best + self.mutation_factor * (x3 - x2), self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant_vector = np.clip(x1 + self.mutation_factor * (x3 - x2), self.lower_bound, self.upper_bound)

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