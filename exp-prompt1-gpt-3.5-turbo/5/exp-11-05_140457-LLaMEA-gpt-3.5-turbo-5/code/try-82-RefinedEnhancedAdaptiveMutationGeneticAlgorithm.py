import numpy as np

class RefinedEnhancedAdaptiveMutationGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15  # Increase population size for better diversity
        self.mutation_prob = 0.2  # Increase mutation probability for more exploration
        self.crossover_rate = 0.8  # Decrease crossover rate for more exploitation

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        mutation_rate = np.full(self.population_size, 0.1)
        
        for _ in range(self.budget):
            fitness = np.array([func(individual) for individual in population])
            fitness_sorted_idx = np.argsort(fitness)
            fittest_individual = population[fitness_sorted_idx[0]]
            
            new_population = [fittest_individual]
            for i in range(1, self.population_size):
                parents_idx = np.random.choice(self.population_size, 5, replace=False)  # Increase parents for better genetic diversity
                parent1, parent2, parent3, parent4, parent5 = population[parents_idx]
                if i == 1:  # Elitism - Keep the best individual
                    child = fittest_individual + mutation_rate[i] * np.random.uniform(-1, 1, self.dim)
                else:
                    child = parent1 + mutation_rate[i] * (parent2 - parent3) + mutation_rate[i] * (parent4 - parent5)  # Incorporate more parents in mutation
                
                if np.random.uniform() < self.mutation_prob:
                    child += np.random.normal(0, 1, self.dim) * (1 - 0.99 * (_ / self.budget))
                
                if np.random.uniform() < self.crossover_rate:
                    cross_points = np.random.choice([True, False], size=self.dim, p=[0.3, 0.7])  # Change crossover strategy to incorporate both parents
                    child[cross_points] = parent1[cross_points]
                    
                new_population.append(child)
                
                # Update mutation rate for each individual
                if i > 1:
                    mutation_rate[i] = np.clip(mutation_rate[i] + np.random.normal(0, 0.1), 0.01, 0.5)
            
            population = np.array(new_population)
        
        best_idx = np.argmin([func(individual) for individual in population])
        return population[best_idx]