import numpy as np

class EnhancedHybridDEAPopulationAdapt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(0.6 * dim)  # Adjusted population size for exploration
        self.prob_crossover = 0.85  # Adjusted crossover probability
        self.F = 0.9  # Increased differential weight for better exploration
        self.current_evaluations = 0
        self.temperature = 2.0  # Higher initial temperature for better exploration
        self.cooling_rate = 0.95  # Adjusted cooling rate
        self.diversity_factor = 0.25  # Increased diversity factor
        self.adaptive_rate = 0.2  # Rate to adapt population size

    def __call__(self, func):
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.current_evaluations < self.budget:
            for i in range(self.population_size):
                if self.current_evaluations >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = indices
                while a == i or b == i or c == i:
                    a, b, c = np.random.choice(self.population_size, 3, replace=False)

                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                if np.random.rand() < self.diversity_factor:
                    direction = np.random.randn(self.dim)
                    direction /= np.linalg.norm(direction)
                    mutant = population[i] + direction * (self.upper_bound - self.lower_bound) * 0.15
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < self.prob_crossover
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                self.current_evaluations += 1
                delta_e = trial_fitness - fitness[i]
                acceptance_probability = np.exp(-delta_e / (self.temperature + 1e-9))

                if trial_fitness < fitness[i] or np.random.rand() < acceptance_probability:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            if self.current_evaluations < self.budget / 2:
                self.population_size = int(self.population_size * (1 + self.adaptive_rate))
                self.population_size = min(self.population_size, self.budget - self.current_evaluations)
                new_population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                population = np.vstack((population, new_population))
                fitness = np.append(fitness, [func(ind) for ind in new_population])
                self.current_evaluations += len(new_population)

            self.temperature *= self.cooling_rate

        return best_solution