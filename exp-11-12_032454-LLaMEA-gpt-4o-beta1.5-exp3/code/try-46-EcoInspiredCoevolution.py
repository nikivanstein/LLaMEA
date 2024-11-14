import numpy as np

class EcoInspiredCoevolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.alpha = 0.5  # degree of competition influence
        self.beta = 0.5  # degree of cooperation influence
        
    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        
        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Find two species for interaction
                indices = list(range(self.population_size))
                indices.remove(i)
                j, k = np.random.choice(indices, 2, replace=False)
                
                # Competition and cooperation dynamics
                if fitness[j] < fitness[k]:
                    interaction_partner = population[j]
                else:
                    interaction_partner = population[k]
                
                # Mutation and crossover inspired by eco-dynamics
                mutated_vector = population[i] + self.alpha * (interaction_partner - population[i])
                mutated_vector = np.clip(mutated_vector, self.lb, self.ub)
                
                # Cooperation for exploration
                cooperative_vector = population[i] + self.beta * (best_individual - population[i])
                cooperative_vector = np.clip(cooperative_vector, self.lb, self.ub)
                
                # Randomly choose between competition and cooperation
                if np.random.rand() < 0.5:
                    offspring = mutated_vector
                else:
                    offspring = cooperative_vector
                
                # Evaluate offspring
                offspring_fitness = func(offspring)
                num_evaluations += 1
                
                # Selection
                if offspring_fitness < fitness[i]:
                    new_population[i] = offspring
                    fitness[i] = offspring_fitness
                    if offspring_fitness < best_fitness:
                        best_individual = offspring
                        best_fitness = offspring_fitness
            population = new_population
        
        return best_individual, best_fitness