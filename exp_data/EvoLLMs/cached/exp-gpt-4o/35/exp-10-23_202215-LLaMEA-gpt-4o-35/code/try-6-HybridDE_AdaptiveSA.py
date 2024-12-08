import numpy as np

class HybridDE_AdaptiveSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 25
        self.mutation_factor = 0.8  # Set mutation factor for DE
        self.crossover_probability = 0.7  # Set crossover probability for DE
        self.initial_temp = 0.9  # Adjusted initial temperature for SA
        self.cooling_rate = 0.98  # Modified cooling rate for slower cooling

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best)
        global_best_idx = np.argmin(personal_best_fitness)
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Differential Evolution Mutation and Crossover
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_probability, mutant_vector, population[i])
                trial_vector = np.clip(trial_vector, self.bounds[0], self.bounds[1])

                # Evaluate trial vector
                fitness = func(trial_vector)
                self.evaluations += 1

                # Update personal best with adaptive SA
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = trial_vector
                    personal_best_fitness[i] = fitness
                else:
                    current_temp = self.initial_temp * (self.cooling_rate ** (self.evaluations / self.budget))
                    acceptance_prob = np.exp((personal_best_fitness[i] - fitness) / (current_temp + 1e-10))
                    if np.random.rand() < acceptance_prob:
                        personal_best[i] = trial_vector
                        personal_best_fitness[i] = fitness

            # Update global best
            global_best_idx = np.argmin(personal_best_fitness)

        return personal_best[global_best_idx], personal_best_fitness[global_best_idx]