import numpy as np

class EnhancedAdaptiveEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 + int(8 * np.log(self.dim))  # Slightly adjusted population size
        self.bounds = (-5.0, 5.0)
        self.scale_factor = 0.85  # Adjusted scale factor for more aggressive mutations
        self.cross_prob = 0.95  # Increased cross probability for diverse trials
        self.adaptation_rate = 0.07  # Adjusted adaptation rate for quicker parameter tuning

    def __call__(self, func):
        # Initialize population within bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        func_evals = self.population_size
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        
        while func_evals < self.budget:
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
                crossover = np.random.rand(self.dim) < (self.cross_prob + adapt_factor)
                trial[crossover] = mutant[crossover]

                # Selection with strategic elitism
                trial_fitness = func(trial)
                func_evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            # Self-adaptive parameter tuning with more granular adjustments
            self.scale_factor = np.random.uniform(0.75, 0.9)  # Slight adjustment
            self.cross_prob = np.random.uniform(0.92, 1.0)    # Slight adjustment

        # Return the best found solution
        return best_solution, best_fitness