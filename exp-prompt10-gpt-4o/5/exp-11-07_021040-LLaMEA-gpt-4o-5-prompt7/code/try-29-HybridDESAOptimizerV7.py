import numpy as np

class HybridDESAOptimizerV7:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(3 * dim, 15)
        self.F = 0.85  # Slightly lower mutation factor for balanced exploration
        self.CR = 0.9  # Increased crossover rate for diversity
        self.initial_temp = 100
        self.cooling_rate = 0.92  # Adjusted cooling rate for smoother annealing

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        while evaluations < self.budget:
            new_population = population.copy()  # Copy current population for updates
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                
                mutant = np.clip(x0 + self.F * (x1 - x2), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                temp = self.initial_temp * (self.cooling_rate ** (evaluations / self.population_size))
                if trial_fitness < best_fitness or np.exp((best_fitness - trial_fitness) / temp) > np.random.rand():
                    best = trial.copy()
                    best_fitness = trial_fitness

            population = new_population

        return best, best_fitness