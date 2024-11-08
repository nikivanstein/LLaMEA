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
        
        # Cache random numbers to reduce calls
        random_state = np.random.RandomState()
        
        # Evolutionary loop
        temperature = self.temp_init
        while evals < self.budget:
            idxs = random_state.choice(self.population_size, (self.population_size, 3), replace=False)
            for i, (idx1, idx2, idx3) in enumerate(idxs):
                # Differential Evolution Mutation
                x1, x2, x3 = population[idx1], population[idx2], population[idx3]
                mutant = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)
                
                # Binomial Crossover
                cross_points = random_state.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                
                # Fitness evaluation
                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                # Simulated Annealing Acceptance
                delta = trial_fitness - fitness[i]
                if delta < 0 or random_state.rand() < np.exp(-delta / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Cooling schedule
            temperature = max(self.temp_final, temperature * self.cooling_rate)
        
        return population[np.argmin(fitness)]