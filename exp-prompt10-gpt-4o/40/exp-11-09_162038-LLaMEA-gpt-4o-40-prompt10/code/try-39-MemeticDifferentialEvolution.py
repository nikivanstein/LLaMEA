import numpy as np

class MemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation and crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                
                # Evaluate trial individual
                trial_fitness = func(trial)
                eval_count += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if eval_count >= self.budget:
                    break

            # Local search on best individual
            best_index = np.argmin(fitness)
            best_individual = population[best_index]
            best_fitness = fitness[best_index]
            
            # Perform a simple random neighborhood search around the best individual
            local_neighbors = best_individual + np.random.uniform(-0.1, 0.1, (10, self.dim))
            local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
            local_fitness = np.array([func(ind) for ind in local_neighbors])
            eval_count += len(local_neighbors)
            
            # Update best if any local neighbor is better
            if np.min(local_fitness) < best_fitness:
                best_index = np.argmin(local_fitness)
                best_individual = local_neighbors[best_index]
                best_fitness = local_fitness[best_index]
                
            population[0] = best_individual
            fitness[0] = best_fitness

        # Return best found solution
        best_index = np.argmin(fitness)
        return population[best_index]