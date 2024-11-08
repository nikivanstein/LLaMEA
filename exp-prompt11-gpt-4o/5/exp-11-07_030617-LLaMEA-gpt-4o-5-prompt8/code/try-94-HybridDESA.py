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
        
        # Precompute random indices for mutation in batches to enhance vectorized operations
        random_indices = np.random.randint(0, self.population_size, (self.budget - evals, 3))
        rand_idx_batches = np.split(random_indices, len(random_indices) // self.population_size)
        
        # Evolutionary loop
        temperature = self.temp_init
        batch_idx = 0
        while evals < self.budget:
            # Use precomputed batch indices to reduce redundant indexing operations
            batch_idxs = rand_idx_batches[batch_idx]
            batch_idx = (batch_idx + 1) % len(rand_idx_batches)
            mutants = population[batch_idxs[:, 0]] + self.F * (population[batch_idxs[:, 1]] - population[batch_idxs[:, 2]])
            mutants = np.clip(mutants, self.lower_bound, self.upper_bound)

            # Binomial Crossover with vectorized operations
            crossover_mask = np.random.rand(self.population_size, self.dim) < self.CR
            trials = np.where(crossover_mask, mutants, population)
            
            # Fitness evaluation and Simulated Annealing Acceptance in a vectorized manner
            trial_fitness = np.array([func(trial) for trial in trials])
            evals += self.population_size
            improved = trial_fitness < fitness
            accept_prob = np.exp(-(trial_fitness - fitness) / temperature)
            accepted = improved | (np.random.rand(self.population_size) < accept_prob)
            population[accepted] = trials[accepted]
            fitness[accepted] = trial_fitness[accepted]

            # Cooling schedule with linear decay
            temperature = self.temp_final + (temperature - self.temp_final) * self.cooling_rate
        
        return population[np.argmin(fitness)]