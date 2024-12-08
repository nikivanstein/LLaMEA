import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.mutation_factor = 0.75 + 0.25 * np.random.rand()  # Adapted mutation factor
        self.crossover_rate = 0.85  # Tweaked crossover rate
        self.initial_temperature = 2.5  # Modified for broader initial search
        self.cooling_schedule = lambda t, i: t * (0.9 + 0.1 * np.cos(i / 7))  # Refined stochastic cooling
        self.cooling_rate = 0.95  # Altered cooling rate
        self.levy_exponent = 1.5  # Introduced Levy flight parameter
        self.uncertainty_weight = 0.4  # New factor for uncertainty-driven mutation

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        temperature = self.initial_temperature

        def levy_flight(size):
            u = np.random.normal(0, 1, size)
            v = np.random.normal(0, 1, size)
            step = u / np.power(np.abs(v), 1 / self.levy_exponent)
            return 0.01 * step * (population - np.mean(population, axis=0))

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                if evaluations % 50 == 0:  # Adjusted frequency for mutation updates
                    self.mutation_factor = 0.55 + 0.35 * np.random.rand() + self.uncertainty_weight * np.random.randn()

                if evaluations % 80 == 0: 
                    best_idx = np.argmin(fitness)
                    x_best = population[best_idx]
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant_vector = np.clip(x_best + self.mutation_factor * (x2 - x3) + levy_flight(self.dim), self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant_vector = np.clip(x1 + self.mutation_factor * (x2 - x3) + levy_flight(self.dim), self.lower_bound, self.upper_bound)

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