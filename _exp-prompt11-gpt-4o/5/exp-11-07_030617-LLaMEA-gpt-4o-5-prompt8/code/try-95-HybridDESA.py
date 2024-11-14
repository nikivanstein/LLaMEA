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
        self.cooling_rate = 0.97  # Adjusted cooling rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size
        
        # Precompute random indices for mutation
        random_indices = np.random.randint(0, self.population_size, (self.budget - evals, 3))
        rand_idx_count = 0

        # Evolutionary loop
        temperature = self.temp_init
        while evals < self.budget:
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                # Differential Evolution Mutation with precomputed indices
                idxs = random_indices[rand_idx_count]
                rand_idx_count += 1
                x1, x2, x3 = population[idxs]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                # Binomial Crossover using boolean mask
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Fitness evaluation
                trial_fitness = func(trial)
                evals += 1

                # Simulated Annealing Acceptance
                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Cooling schedule with linear decay
            temperature = self.temp_final + (temperature - self.temp_final) * self.cooling_rate
        
        return population[np.argmin(fitness)]