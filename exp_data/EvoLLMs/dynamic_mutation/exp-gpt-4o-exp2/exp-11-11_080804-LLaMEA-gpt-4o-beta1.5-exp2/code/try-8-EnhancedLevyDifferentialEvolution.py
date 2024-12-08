import numpy as np

class EnhancedLevyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f = 0.5
        self.cr = 0.9
        self.alpha = 1.5

    def levy_flight(self, size):
        u = np.random.normal(0, 1, size) * (np.sqrt(np.abs(np.random.normal(0, 1, size))) ** (-1 / self.alpha))
        return u

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation: DE/rand/1
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.f * (x1 - x2), self.lower_bound, self.upper_bound)
                
                # Dynamic Crossover
                crossover_prob = np.random.rand()
                crossover = np.random.rand(self.dim) < (self.cr * crossover_prob)
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])
                
                # Enhanced Levy flight with adaptive step size
                step_size = (self.budget - evaluations) / self.budget
                levy_step = self.levy_flight(self.dim)
                trial += step_size * levy_step * np.exp(-evaluations / self.budget)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                
                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best = trial

                if evaluations >= self.budget:
                    break

        return best