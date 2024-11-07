import numpy as np

class CrowdedRefinedEnhancedAdaptiveMutationGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.mutation_prob = 0.1
        self.crossover_rate = 0.9

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        mutation_rate = np.full(self.population_size, 0.1)
        
        for _ in range(self.budget):
            fitness = np.array([func(individual) for individual in population])
            fitness_sorted_idx = np.argsort(fitness)
            fittest_individual = population[fitness_sorted_idx[0]]
            
            new_population = [fittest_individual]
            for i in range(1, self.population_size):
                parents_idx = np.random.choice(self.population_size, 3, replace=False)
                parent1, parent2, parent3 = population[parents_idx]
                if i == 1:  # Elitism - Keep the best individual
                    child = fittest_individual + mutation_rate[i] * np.random.uniform(-1, 1, self.dim)
                else:
                    child = parent1 + mutation_rate[i] * (parent2 - parent3)
                    
                if np.random.uniform() < self.mutation_prob:
                    child += np.random.normal(0, 1, self.dim) * (1 - 0.99 * (_ / self.budget))
                
                if np.random.uniform() < self.crossover_rate:
                    cross_points = np.random.randint(0, 2, self.dim).astype(bool)
                    child[cross_points] = parent1[cross_points]
                    
                new_population.append(child)
                
                # Update mutation rate for each individual
                if i > 1:
                    mutation_rate[i] = np.clip(mutation_rate[i] + np.random.normal(0, 0.1), 0.01, 0.5)
            
            population = np.array(new_population)
            
            # Calculate crowding distance for diversity maintenance
            crowding_dist = np.zeros(self.population_size)
            for j in range(self.dim):
                sorted_fitness_idx = np.argsort([func(individual) for individual in population])
                crowding_dist[sorted_fitness_idx[0]] = np.inf
                crowding_dist[sorted_fitness_idx[-1]] = np.inf
                sorted_fitness = fitness[sorted_fitness_idx]
                crowding_dist[sorted_fitness_idx[1:-1]] += (sorted_fitness[2:] - sorted_fitness[:-2]) / (sorted_fitness[-1] - sorted_fitness[0])
            
            population = population[np.argsort(-crowding_dist)]
        
        best_idx = np.argmin([func(individual) for individual in population])
        return population[best_idx]