import numpy as np

class EnhancedHybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(10 * np.log(self.dim))
        self.bounds = (-5.0, 5.0)
        self.scale_factor = 0.8
        self.cross_prob = 0.9
        self.adaptation_rate = 0.05
        self.local_search_prob = 0.1  # New parameter for local search

    def __call__(self, func):
        # Initialize population within bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        func_evals = self.population_size
        
        while func_evals < self.budget:
            for i in range(self.population_size):
                if func_evals >= self.budget:
                    break

                # Mutation: choose three random indices different from i
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Adaptive Differential Mutation
                adapt_factor = self.adaptation_rate * (np.random.rand() - 0.5)
                mutant = a + (self.scale_factor + adapt_factor) * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Dynamic Crossover
                trial = np.copy(population[i])
                crossover = np.random.rand(self.dim) < (self.cross_prob + adapt_factor)
                trial[crossover] = mutant[crossover]

                # Local Search Intensification
                if np.random.rand() < self.local_search_prob:
                    neighborhood = population[np.random.choice(idxs, 2, replace=False)]
                    local_search_point = neighborhood.mean(axis=0)
                    local_search_point = np.clip(local_search_point + 0.1 * np.random.randn(self.dim), self.bounds[0], self.bounds[1])
                    local_search_fitness = func(local_search_point)
                    func_evals += 1
                    if local_search_fitness < trial_fitness:
                        trial = local_search_point
                        trial_fitness = local_search_fitness

                # Selection
                trial_fitness = func(trial)
                func_evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Self-adaptive parameter tuning
            self.scale_factor = np.random.uniform(0.6, 0.85)
            self.cross_prob = np.random.uniform(0.85, 1.0)

        # Return the best found solution
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]