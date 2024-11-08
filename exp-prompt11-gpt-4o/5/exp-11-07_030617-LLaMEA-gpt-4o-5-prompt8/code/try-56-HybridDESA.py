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
        
        # Precompute random indices for mutation
        random_indices = np.random.randint(0, self.population_size, (self.budget - evals, 3))
        rand_idx_count = 0

        # Evolutionary loop with vectorized operations
        temperature = self.temp_init
        while evals < self.budget:
            # Vectorize mutation and crossover operations to reduce loop overhead
            idxs = random_indices[rand_idx_count:rand_idx_count+self.population_size]
            rand_idx_count += self.population_size
            x1, x2, x3 = population[idxs[:, 0]], population[idxs[:, 1]], population[idxs[:, 2]]
            mutants = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)
            crossover_masks = np.random.rand(self.population_size, self.dim) < self.CR
            trials = np.where(crossover_masks, mutants, population)
            
            # Evaluate fitness and apply simulated annealing acceptance
            trial_fitnesses = np.array([func(trial) for trial in trials])
            evals += self.population_size
            delta = trial_fitnesses - fitness
            acceptance_probs = np.exp(-delta / temperature)
            acceptance = (delta < 0) | (np.random.rand(self.population_size) < acceptance_probs)
            population[acceptance] = trials[acceptance]
            fitness[acceptance] = trial_fitnesses[acceptance]

            # Update temperature
            temperature = max(self.temp_final, temperature * self.cooling_rate)
        
        return population[np.argmin(fitness)]