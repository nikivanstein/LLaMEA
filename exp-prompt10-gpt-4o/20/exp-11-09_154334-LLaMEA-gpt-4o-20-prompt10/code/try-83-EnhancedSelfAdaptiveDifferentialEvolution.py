import numpy as np

class EnhancedSelfAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim  # Further reduced population for rapid convergence
        self.initial_mutation_factor = 0.9  # Increased initial mutation factor for better exploration
        self.crossover_rate = 0.85  # Adjusted crossover rate for improved selection
        self.mutation_decay = 0.95  # More aggressive decay for mutation factor
        self.split_factor = 0.4  # Adjusted subpopulation split factor
        self.dynamic_pop_resize = True  # Flag for dynamic population resizing

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
                sample_size = subpop_size if np.random.rand() < 0.5 else self.population_size
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                elif np.random.rand() < 0.1:  # Greedy selection pressure
                    population[i] = trial  # Occasionally accept worse solutions

                if evals >= self.budget:
                    break

            mutation_factor *= self.mutation_decay
            if evals % (self.population_size // 2) == 0:
                self.crossover_rate = 0.7 + 0.3 * np.random.rand()

            if self.dynamic_pop_resize and evals % (self.budget // 4) == 0 and evals < self.budget // 2:
                self.population_size = int(self.population_size * 0.8)  # Reduce population size adaptively
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]

        best_index = np.argmin(fitness)
        return population[best_index]