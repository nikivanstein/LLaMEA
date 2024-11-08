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
        self.cooling_rate = 0.95  # Further adjusted cooling rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        # Precompute random indices for mutation using numpy matrix operation
        random_indices = np.random.randint(0, self.population_size, (self.population_size, 3))
        rand_idx_count = 0

        # Evolutionary loop
        temperature = self.temp_init
        while evals < self.budget:
            # Vectorized loop to enhance efficiency
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                # Differential Evolution Mutation using vectorized approach
                idxs = random_indices[i]
                x1, x2, x3 = population[idxs]
                mutant = x1 + self.F * (x2 - x3)
                np.clip(mutant, self.lower_bound, self.upper_bound, out=mutant)
                
                # Binomial Crossover using numpy's broadcasting
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Fitness evaluation
                trial_fitness = func(trial)
                evals += 1

                # Simulated Annealing Acceptance
                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Cooling schedule with exponential decay
            temperature *= self.cooling_rate
        
        return population[np.argmin(fitness)]