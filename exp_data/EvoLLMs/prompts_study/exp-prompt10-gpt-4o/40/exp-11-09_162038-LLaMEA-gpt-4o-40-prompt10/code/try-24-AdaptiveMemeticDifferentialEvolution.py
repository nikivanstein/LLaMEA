import numpy as np

class AdaptiveMemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.8  # Initial differential weight
        self.CR = 0.9  # Crossover probability
        self.elitism_rate = 0.2  # Rate of elite individuals for local search

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        
        while eval_count < self.budget:
            # Sort population by fitness
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            # Adaptive differential evolution
            for i in range(self.population_size):
                if np.random.rand() < 0.5:  # Adapt F dynamically
                    self.F = 0.5 * np.random.rand() + 0.5
                
                # Mutation and crossover with elitism
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

            # Elitist local search on top individuals
            num_elites = int(self.elitism_rate * self.population_size)
            for j in range(num_elites):
                elite_individual = population[j]
                elite_fitness = fitness[j]
                
                # Random local search around elite individual
                local_neighbors = elite_individual + np.random.uniform(-0.1, 0.1, (5, self.dim))
                local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
                local_fitness = np.array([func(ind) for ind in local_neighbors])
                eval_count += len(local_neighbors)
                
                # Update elite if any local neighbor is better
                if np.min(local_fitness) < elite_fitness:
                    best_index = np.argmin(local_fitness)
                    population[j] = local_neighbors[best_index]
                    fitness[j] = local_fitness[best_index]

        # Return best found solution
        best_index = np.argmin(fitness)
        return population[best_index]