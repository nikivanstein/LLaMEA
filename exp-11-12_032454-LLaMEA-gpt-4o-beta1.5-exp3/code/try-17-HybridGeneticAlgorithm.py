import numpy as np

class HybridGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(10, int(budget / (15 * dim)))  # heuristic for population size
        self.mutation_scale = 0.5  # scale for differential mutation
        self.crossover_rate = 0.7  # probability of crossover
        self.adaptive_topology_rate = 0.2  # fraction of population to adaptively change topology

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Select parents for crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices[0]], population[indices[1]], population[indices[2]]
                
                # Differential mutation and crossover
                mutant = a + self.mutation_scale * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, population[i])
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                num_evaluations += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
            
            # Adaptive Topology Change
            num_topology_change = int(self.adaptive_topology_rate * self.population_size)
            worst_indices = np.argsort(fitness)[-num_topology_change:]
            for i in worst_indices:
                if num_evaluations >= self.budget:
                    break
                new_population[i] = np.random.uniform(self.lb, self.ub, self.dim)
                fitness[i] = func(new_population[i])
                num_evaluations += 1
            
            population = new_population
        
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]