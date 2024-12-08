import numpy as np

class EnhancedHybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + int(10 * np.log(self.dim))
        self.bounds = (-5.0, 5.0)
        self.scale_factor = 0.85
        self.cross_prob = 0.9
        self.adaptation_rate = 0.05
        self.exploit_factor = 0.2

    def __call__(self, func):
        population_size = self.initial_population_size
        population = np.random.uniform(self.bounds[0], self.bounds[1], (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        func_evals = population_size
        
        while func_evals < self.budget:
            for i in range(population_size):
                if func_evals >= self.budget:
                    break

                idxs = list(range(population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                adapt_factor = self.adaptation_rate * (np.random.rand() - 0.5)
                exploit_factor = self.exploit_factor * (0.5 - np.abs(np.mean(fitness) - fitness[i]) / np.std(fitness))
                dynamic_scale = self.scale_factor + adapt_factor + exploit_factor
                mutant = a + dynamic_scale * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                trial = np.copy(population[i])
                crossover = np.random.rand(self.dim) < (self.cross_prob + adapt_factor * np.random.uniform(0.9, 1.05))
                trial[crossover] = mutant[crossover]

                trial_fitness = func(trial)
                func_evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            self.scale_factor = np.random.uniform(0.75, 0.85)
            self.cross_prob = np.random.uniform(0.88, 1.0)

            if np.std(fitness) < 1e-5:
                population_size = max(4, int(population_size * 0.9))
                idxs = np.argsort(fitness)[:population_size]
                population = population[idxs]
                fitness = fitness[idxs]

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]