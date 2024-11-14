import numpy as np

class EnhancedSelfAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim  # Reduced population size for faster convergence
        self.initial_mutation_factor = 0.9  # Increased initial mutation factor for better exploration
        self.crossover_rate = 0.85  # Slightly reduced crossover rate for enhanced exploitation
        self.mutation_decay = 0.90  # Increased decay for more aggressive convergence
        self.split_factor = 0.6  # Increased subpopulation split factor

    def __call__(self, func):
        np.random.seed(0)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evals = self.population_size
        mutation_factor = self.initial_mutation_factor
        stagnation_counter = 0
        best_fitness = np.min(fitness)

        while evals < self.budget:
            subpop_size = int(self.population_size * self.split_factor)
            for i in range(self.population_size):
                idxs = list(range(self.population_size))
                idxs.remove(i)
                sample_size = subpop_size if np.random.rand() < 0.6 else self.population_size  # More dynamic subpopulation
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial

                if evals >= self.budget:
                    break

            new_best_fitness = np.min(fitness)
            if new_best_fitness < best_fitness:
                best_fitness = new_best_fitness
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > self.population_size // 3:
                mutation_factor *= 1.05  # Adaptively increase mutation factor in stagnation
                stagnation_counter = 0

            mutation_factor *= self.mutation_decay  # Apply decay to mutation factor
            if evals % (self.population_size // 2) == 0:  # Adapt learning rate dynamically
                self.crossover_rate = 0.75 + 0.25 * np.random.rand()

        best_index = np.argmin(fitness)
        return population[best_index]