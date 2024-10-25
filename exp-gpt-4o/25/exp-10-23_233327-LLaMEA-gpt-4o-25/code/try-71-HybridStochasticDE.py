import numpy as np

class HybridStochasticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 25  # Increased population size for diversity
        self.mutation_factor = 0.7 + 0.4 * np.random.rand()  # Increased mutation factor range
        self.crossover_rate = 0.8  # Slightly reduced crossover rate to encourage exploration
        self.initial_temperature = 3.0  # Higher initial temperature for broad initial search
        self.cooling_schedule = lambda t, i: t * (0.9 + 0.05 * np.cos(i / 10))  # Dynamic cooling schedule
        self.cooling_rate = 0.85  # Slower cooling rate for sustained exploration
        self.levy_factor = 1.5  # Levy flight factor for mutation

    def levy_flight(self, size):
        u = np.random.normal(0, 1, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / self.levy_factor)
        return step

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

                if evaluations % 40 == 0:  # More frequent dynamic mutation
                    self.mutation_factor = 0.5 + 0.3 * np.random.rand() + self.levy_flight(1)[0]

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