import numpy as np

class AdaptiveMemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_min, self.F_max = 0.5, 1.0  # Adaptive differential weight range
        self.CR_min, self.CR_max = 0.1, 0.9  # Adaptive crossover probability range

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        stagnation_counter = 0
        best_fitness = np.min(fitness)
        
        while eval_count < self.budget:
            if stagnation_counter >= 5:  # adaptive adjustment if no improvement
                self.F_min, self.F_max = 0.6, 1.2
                self.CR_min, self.CR_max = 0.2, 0.8
                stagnation_counter = 0
            
            for i in range(self.population_size):
                # Adaptive mutation factor and crossover probability
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)

                # Mutation and crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                
                # Evaluate trial individual
                trial_fitness = func(trial)
                eval_count += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        stagnation_counter = 0
                
                if eval_count >= self.budget:
                    break

            # Enhanced local search on best individual
            best_index = np.argmin(fitness)
            best_individual = population[best_index]
            
            # Perform a more extensive random neighborhood search around the best individual
            local_neighbors = best_individual + np.random.normal(0, 0.1, (20, self.dim))
            local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
            local_fitness = np.array([func(ind) for ind in local_neighbors])
            eval_count += len(local_neighbors)
            
            # Update best if any local neighbor is better
            if np.min(local_fitness) < best_fitness:
                best_index = np.argmin(local_fitness)
                best_individual = local_neighbors[best_index]
                best_fitness = local_fitness[best_index]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            population[0] = best_individual
            fitness[0] = best_fitness

        # Return best found solution
        best_index = np.argmin(fitness)
        return population[best_index]