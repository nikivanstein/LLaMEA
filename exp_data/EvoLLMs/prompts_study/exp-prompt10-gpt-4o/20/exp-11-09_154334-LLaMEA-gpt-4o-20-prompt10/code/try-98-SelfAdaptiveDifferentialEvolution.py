import numpy as np

class SelfAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim  # Slightly reduced population size for faster convergence
        self.initial_mutation_factor = 0.8  # Adjusted initial mutation factor for exploration
        self.crossover_rate = 0.9  # Increased crossover rate for diversity
        self.mutation_decay = 0.98  # Optimized decay for mutation factor
        self.split_factor = 0.5  # Subpopulation split factor

    def __call__(self, func):
        np.random.seed(0)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evals = self.population_size
        mutation_factor = self.initial_mutation_factor

        while evals < self.budget:
            subpop_size = int(self.population_size * self.split_factor)
            for i in range(self.population_size):
                idxs = list(range(self.population_size))
                idxs.remove(i)
                sample_size = subpop_size if np.random.rand() < 0.5 else self.population_size  # Dynamic subpopulation
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

            mutation_factor *= self.mutation_decay  # Apply decay to mutation factor
            if evals % (self.population_size // 2) == 0:  # Adapt learning rate dynamically
                self.crossover_rate = 0.7 + 0.3 * np.random.rand()

        best_index = np.argmin(fitness)
        return population[best_index]