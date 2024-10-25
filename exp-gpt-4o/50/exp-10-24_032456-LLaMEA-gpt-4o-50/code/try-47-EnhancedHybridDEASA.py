import numpy as np

class EnhancedHybridDEASA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20 + int(0.6 * dim)  # Adjusted population size for diversity and balance
        self.prob_crossover = 0.85  # Balanced crossover probability for better stability
        self.F = 0.7  # Fine-tuned differential weight for effective mutation
        self.current_evaluations = 0
        self.initial_temperature = 2.0  # Higher initial temperature for hill climbing
        self.cooling_rate = 0.85  # Adaptive cooling rate for controlled annealing
        self.diversity_factor = 0.15  # Balanced diversity factor

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        temperature = self.initial_temperature

        while self.current_evaluations < self.budget:
            for i in range(self.population_size):
                if self.current_evaluations >= self.budget:
                    break

                # Select three random individuals for mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = indices
                while a == i or b == i or c == i:
                    a, b, c = np.random.choice(self.population_size, 3, replace=False)

                # Mutate with differential evolution strategy
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Introduce diversity via stochastic hill climbing
                if np.random.rand() < self.diversity_factor:
                    direction = np.random.randn(self.dim)
                    direction /= np.linalg.norm(direction)
                    mutant = population[i] + direction * np.random.uniform(0.05, 0.15) * (self.upper_bound - self.lower_bound)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.prob_crossover
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate new solution
                trial_fitness = func(trial)
                self.current_evaluations += 1

                # Metropolis criterion for acceptance
                delta_e = trial_fitness - fitness[i]
                acceptance_probability = np.exp(-delta_e / (temperature + 1e-9))
                temperature *= self.cooling_rate  # Adjust temperature using adaptive cooling

                if trial_fitness < fitness[i] or np.random.rand() < acceptance_probability:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update the best solution found
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

        return best_solution