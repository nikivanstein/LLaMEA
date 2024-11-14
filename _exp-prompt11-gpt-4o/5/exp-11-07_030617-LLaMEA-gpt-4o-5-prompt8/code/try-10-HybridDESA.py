import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.temp_init = 100  # Initial temperature for simulated annealing
        self.temp_final = 0.1  # Final temperature
        self.cooling_rate = 0.99  # Cooling rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population and fitness
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size
        
        # Evolutionary loop
        temperature = self.temp_init
        while evals < self.budget:
            for i in range(self.population_size):
                # Differential Evolution Mutation
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[idxs]
                mutant = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)
                
                # Binomial Crossover
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                
                # Fitness evaluation only if necessary
                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                # Simulated Annealing Acceptance
                delta = trial_fitness - fitness[i]
                if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Cooling schedule
            temperature *= self.cooling_rate
            if temperature < self.temp_final:
                temperature = self.temp_final
        
        return population[np.argmin(fitness)]