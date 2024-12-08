import numpy as np

class AdaptiveMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))
        self.F = 0.5
        self.CR = 0.9
        self.local_search_prob = 0.1  # Probability of applying local search

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutated_vector = population[a] + self.F * (population[b] - population[c])
                mutated_vector = np.clip(mutated_vector, self.lb, self.ub)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                offspring = np.where(crossover_mask, mutated_vector, population[i])
                
                # Evaluate offspring
                offspring_fitness = func(offspring)
                num_evaluations += 1
                
                # Selection
                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness
                    if offspring_fitness < best_fitness:
                        best_individual = offspring
                        best_fitness = offspring_fitness
                
                # Local optimization with dynamic probability
                if np.random.rand() < self.local_search_prob:
                    local_best, local_fitness = self.local_search(offspring, func)
                    num_evaluations += 1  # Local search uses one function evaluation
                    if local_fitness < fitness[i]:
                        population[i] = local_best
                        fitness[i] = local_fitness
                        if local_fitness < best_fitness:
                            best_individual = local_best
                            best_fitness = local_fitness
        
        return best_individual, best_fitness

    def local_search(self, individual, func):
        # Simple random search in neighborhood
        perturbation = np.random.uniform(-0.1, 0.1, size=self.dim)
        candidate = np.clip(individual + perturbation, self.lb, self.ub)
        candidate_fitness = func(candidate)
        if candidate_fitness < func(individual):
            return candidate, candidate_fitness
        return individual, func(individual)