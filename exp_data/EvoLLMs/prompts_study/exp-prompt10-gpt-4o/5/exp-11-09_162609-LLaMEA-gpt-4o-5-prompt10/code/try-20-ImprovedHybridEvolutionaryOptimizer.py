import numpy as np

class ImprovedHybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(10 * np.log(self.dim))
        self.bounds = (-5.0, 5.0)
        self.scale_factor = 0.8
        self.cross_prob = 0.9
        self.adaptation_rate = 0.05

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        func_evals = self.population_size
        
        while func_evals < self.budget:
            for i in range(self.population_size):
                if func_evals >= self.budget:
                    break

                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                adapt_factor = self.adaptation_rate * (np.random.rand() - 0.5)
                mutant = a + (self.scale_factor + adapt_factor) * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
                
                # Enhanced Diversity Strategy
                if np.random.rand() < 0.2:
                    mutant += np.random.normal(0, 0.1, self.dim)  # Small random perturbation
                
                trial = np.copy(population[i])
                crossover = np.random.rand(self.dim) < (self.cross_prob + adapt_factor)
                trial[crossover] = mutant[crossover]

                trial_fitness = func(trial)
                func_evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Refined Self-adaptive Parameter Tuning
            self.scale_factor = np.random.uniform(0.65, 0.85)
            self.cross_prob = np.random.uniform(0.88, 1.0)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]