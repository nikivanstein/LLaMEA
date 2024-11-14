import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Increased population size for diversity
        self.initial_mutation_factor = 0.6  # Lower initial mutation factor for exploitation
        self.crossover_rate = 0.7  # Lower crossover rate to retain strong individuals
        self.mutation_decay = 0.95  # Slower decay for sustained exploration
        self.split_factor = 0.6  # Increased subpopulation split for stability

    def __call__(self, func):
        np.random.seed(0)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evals = self.population_size
        mutation_factor = self.initial_mutation_factor
        adaptive_crossover_rate = self.crossover_rate

        while evals < self.budget:
            subpop_size = int(self.population_size * self.split_factor)
            for i in range(self.population_size):
                idxs = list(range(self.population_size))
                idxs.remove(i)
                sample_size = subpop_size if np.random.rand() < 0.5 else self.population_size  # Dynamic subpopulation
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < adaptive_crossover_rate, mutant, population[i])
                
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    adaptive_crossover_rate = min(1.0, adaptive_crossover_rate + 0.05)  # Increase crossover rate if successful
                else:
                    adaptive_crossover_rate = max(0.1, adaptive_crossover_rate - 0.02)  # Decrease crossover rate if not successful

                if evals >= self.budget:
                    break

            mutation_factor *= self.mutation_decay  # Apply decay to mutation factor

        best_index = np.argmin(fitness)
        return population[best_index]