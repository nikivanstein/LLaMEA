import numpy as np

class HybridGASA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (12 * dim)))
        self.selection_pressure = 0.2
        self.mutation_rate = 0.1
        self.initial_temperature = 100
        self.cooling_rate = 0.99

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        
        temperature = self.initial_temperature
        
        while num_evaluations < self.budget:
            # Selection
            num_parents = int(self.selection_pressure * self.population_size)
            best_indices = np.argsort(fitness)[:num_parents]
            parents = population[best_indices]
            
            # Crossover
            offspring = []
            while len(offspring) < self.population_size - num_parents:
                parents_indices = np.random.choice(num_parents, 2, replace=False)
                parent1, parent2 = parents[parents_indices]
                crossover_point = np.random.randint(1, self.dim)
                child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                offspring.append(child)
            
            # Mutation
            offspring = np.array(offspring)
            mutation_mask = np.random.rand(*offspring.shape) < self.mutation_rate
            mutation_values = np.random.uniform(self.lb, self.ub, offspring.shape)
            offspring[mutation_mask] = mutation_values[mutation_mask]
            
            # Create new population
            population = np.concatenate((parents, offspring))
            population = np.clip(population, self.lb, self.ub)
            
            # Evaluate new solutions
            fitness = np.array([func(ind) for ind in population])
            num_evaluations += len(population)
            
            # Simulated Annealing Local Search
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                candidate = population[i] + np.random.normal(0, 0.1, self.dim)
                candidate = np.clip(candidate, self.lb, self.ub)
                candidate_fitness = func(candidate)
                num_evaluations += 1
                acceptance_prob = np.exp((fitness[i] - candidate_fitness) / (temperature + 1e-10))
                if candidate_fitness < fitness[i] or np.random.rand() < acceptance_prob:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_solution = candidate
                        best_fitness = candidate_fitness
            
            temperature *= self.cooling_rate
        
        return best_solution, best_fitness