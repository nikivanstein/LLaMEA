import numpy as np

class QuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (15 * dim)))  # heuristic for population size
        self.f = 0.5  # scaling factor
        self.cr = 0.9  # crossover rate
        self.qr = 0.1  # quantum rotation rate

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_index = np.argmin(fitness)
        best = population[best_index]
        best_fitness = fitness[best_index]
        
        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                
                # Differential mutation
                mutant = x0 + self.f * (x1 - x2)
                mutant = np.clip(mutant, self.lb, self.ub)
                
                # Quantum-inspired crossover
                crossover_mask = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Quantum rotation
                if np.random.rand() < self.qr:
                    rotation = np.random.uniform(-np.pi, np.pi, self.dim)
                    trial += np.sin(rotation) * (best - trial)
                    trial = np.clip(trial, self.lb, self.ub)
                
                # Selection
                trial_fitness = func(trial)
                num_evaluations += 1
                
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness
                
                if num_evaluations >= self.budget:
                    break
            
            population = new_population
        
        return best, best_fitness