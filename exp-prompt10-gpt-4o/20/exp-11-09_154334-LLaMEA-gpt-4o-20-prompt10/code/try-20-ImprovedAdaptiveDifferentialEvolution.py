import numpy as np

class ImprovedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        # Dynamic mutation factor and crossover rate
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7

    def __call__(self, func):
        np.random.seed(0)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evals = self.population_size

        # Diversity preservation parameters
        min_diversity_threshold = 1e-5
        max_generation_stagnation = 5
        stagnation_counter = 0
        last_best_fitness = np.inf

        while evals < self.budget:
            for i in range(self.population_size):
                # Adjust mutation factor and crossover rate dynamically
                self.mutation_factor = 0.5 + 0.3 * (1 - evals / self.budget)
                self.crossover_rate = 0.7 + 0.2 * (1 - evals / self.budget)
                
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial

                if evals >= self.budget:
                    break
            
            # Check diversity to prevent stagnation
            current_best_fitness = np.min(fitness)
            if current_best_fitness >= last_best_fitness - min_diversity_threshold:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter >= max_generation_stagnation:
                # Reinitialize a portion of the population to explore new areas
                replace_count = self.population_size // 5
                reinit_indices = np.random.choice(self.population_size, replace_count, replace=False)
                for idx in reinit_indices:
                    population[idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    fitness[idx] = func(population[idx])
                    evals += 1
            
            last_best_fitness = current_best_fitness

        best_index = np.argmin(fitness)
        return population[best_index]