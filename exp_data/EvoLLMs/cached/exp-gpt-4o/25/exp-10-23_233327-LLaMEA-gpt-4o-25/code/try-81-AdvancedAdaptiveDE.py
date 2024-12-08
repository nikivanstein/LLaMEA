import numpy as np

class AdvancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 25  # Slightly increased population size for diversity
        self.mutation_factor = 0.9 * np.random.rand()  # Increased adaptive mutation factor range
        self.crossover_rate = 0.85  # Adjusted crossover rate for balance
        self.initial_temperature = 2.5  # Further increased temperature for exploration
        self.cooling_schedule = lambda t, i: t * (0.88 + 0.07 * np.cos(i / 5))  # Altered cooling schedule
        self.cooling_rate = 0.93  # Adjusted cooling rate for prolonged exploration
        self.chaos_factor = 0.30  # Slightly reduced chaos influence

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

                if evaluations % 47 == 0:  # Different frequency for dynamic mutation
                    self.mutation_factor = 0.5 + 0.4 * np.random.rand() + self.chaos_factor * np.sin(evaluations / 10)

                if evaluations % 75 == 0:  # Modified adaptive mutation condition
                    best_idx = np.argmin(fitness)
                    x_best = population[best_idx]
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant_vector = np.clip(x_best + self.mutation_factor * (x2 - x3), self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    quantum_perturbation = np.random.normal(0, 1, self.dim) * 0.01  # Quantum-inspired perturbation
                    mutant_vector = np.clip(x1 + self.mutation_factor * (x2 - x3) + quantum_perturbation, self.lower_bound, self.upper_bound)

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
                    if np.random.rand() < 0.5 * prob_accept:  # Enhanced probability factor for acceptance
                        population[i] = trial_vector
                        fitness[i] = trial_fitness

            temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]