import numpy as np

class EnhancedMemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_F = 0.8  # Initial differential weight
        self.initial_CR = 0.9  # Initial crossover probability
        self.epsilon = 1e-8  # Small value to prevent division by zero

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        F = self.initial_F
        CR = self.initial_CR
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                # Adaptive mutation and crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant_vector = a + F * (b - c)
                mutant = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                
                trial_vector = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                
                # Evaluate trial individual
                trial_fitness = func(trial_vector)
                eval_count += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                
                if eval_count >= self.budget:
                    break

            # Adaptive parameters update
            F = 0.5 + 0.3 * np.tanh(eval_count / self.budget)
            CR = 0.5 + 0.4 * np.tanh(eval_count / self.budget)

            # Dynamic population adjustment based on diversity
            diversity = np.std(fitness) / (np.mean(fitness) + self.epsilon)
            if diversity < 0.05 and self.population_size > 5:
                self.population_size = max(5, self.population_size - 1)
            elif diversity > 0.1 and self.population_size < min(50, self.budget // 10):
                self.population_size += 1
            
            # Local search on best individual
            best_index = np.argmin(fitness)
            best_individual = population[best_index]
            best_fitness = fitness[best_index]
            
            # Perform an enhanced local search
            local_neighbors = best_individual + np.random.uniform(-0.1, 0.1, (min(10, self.budget - eval_count), self.dim))
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