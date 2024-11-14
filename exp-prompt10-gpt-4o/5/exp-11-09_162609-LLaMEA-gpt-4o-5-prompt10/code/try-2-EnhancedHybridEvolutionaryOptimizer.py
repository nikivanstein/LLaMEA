import numpy as np

class EnhancedHybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + int(10 * np.log(self.dim))
        self.bounds = (-5.0, 5.0)
        self.scale_factor = 0.8
        self.cross_prob = 0.9
        
    def __call__(self, func):
        # Initialize population within bounds
        population_size = self.initial_population_size
        population = np.random.uniform(self.bounds[0], self.bounds[1], (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        func_evals = population_size
        
        while func_evals < self.budget:
            for i in range(population_size):
                if func_evals >= self.budget:
                    break

                # Mutation: choose three random indices different from i
                idxs = list(range(population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Differential Mutation
                mutant = a + self.scale_factor * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Adaptive Crossover
                trial = np.copy(population[i])
                crossover_rate = np.random.uniform(self.cross_prob - 0.05, self.cross_prob + 0.05)
                crossover = np.random.rand(self.dim) < crossover_rate
                trial[crossover] = mutant[crossover]

                # Selection
                trial_fitness = func(trial)
                func_evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Dynamic population size adjustment
            if func_evals < self.budget * 0.5:
                population_size = min(int(1.2 * population_size), (self.budget - func_evals) // 2)
            else:
                population_size = max(int(0.8 * population_size), 4)
            population = np.vstack((population, np.random.uniform(self.bounds[0], self.bounds[1], (population_size - len(population), self.dim))))
            fitness = np.append(fitness, [func(ind) for ind in population[len(fitness):]])
            func_evals += population_size - len(fitness)
            fitness = fitness[:population_size]
            population = population[:population_size]

            # Self-adaptive parameter tuning
            self.scale_factor = np.random.uniform(0.5, 0.9)
            self.cross_prob = np.random.uniform(0.8, 1.0)

        # Return the best found solution
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]