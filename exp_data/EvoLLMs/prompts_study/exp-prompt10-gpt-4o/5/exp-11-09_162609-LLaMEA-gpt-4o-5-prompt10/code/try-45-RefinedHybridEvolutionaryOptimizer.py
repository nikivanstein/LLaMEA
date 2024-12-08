import numpy as np

class RefinedHybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + int(10 * np.log(self.dim))
        self.bounds = (-5.0, 5.0)
        self.scale_factor = 0.75
        self.cross_prob = 0.85
        self.adaptation_rate = 0.07  # Adjusted adaptation rate
        self.population_size = self.initial_population_size
        
    def __call__(self, func):
        # Initialize population within bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        func_evals = self.population_size
        stagnation_counter = 0
        
        while func_evals < self.budget:
            if stagnation_counter > 5:  # Adaptive population resizing
                new_individuals = np.random.uniform(self.bounds[0], self.bounds[1], (2, self.dim))
                population = np.vstack((population, new_individuals))
                fitness = np.append(fitness, [func(ind) for ind in new_individuals])
                func_evals += 2
                self.population_size += 2
                stagnation_counter = 0
            
            prev_best_fitness = np.min(fitness)
            
            for i in range(self.population_size):
                if func_evals >= self.budget:
                    break

                # Mutation: choose three random indices different from i
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Refined Adaptive Differential Mutation
                adapt_factor = self.adaptation_rate * (np.random.rand() - 0.5)
                mutant = a + (self.scale_factor + adapt_factor) * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Dynamic Crossover with enhanced probability adaptation
                trial = np.copy(population[i])
                crossover = np.random.rand(self.dim) < (self.cross_prob + adapt_factor * np.random.uniform(0.9, 1.1))
                trial[crossover] = mutant[crossover]

                # Selection
                trial_fitness = func(trial)
                func_evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Check for convergence stagnation
            current_best_fitness = np.min(fitness)
            if np.isclose(current_best_fitness, prev_best_fitness, rtol=1e-4):
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            # Self-adaptive parameter tuning with more granular adjustments
            self.scale_factor = np.random.uniform(0.7, 0.8)  # Slight adjustment
            self.cross_prob = np.random.uniform(0.83, 0.9)   # Slight adjustment

        # Return the best found solution
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]