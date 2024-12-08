import numpy as np

class AdaptiveBiogeographyWithLevyFlights:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.mutation_probability = 0.1
        self.elite_ratio = 0.2
        
    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=L)
        v = np.random.normal(0, 1, size=L)
        step = u / np.abs(v)**(1 / beta)
        return step

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        
        while num_evaluations < self.budget:
            # Sort population by fitness
            sorted_indices = np.argsort(fitness)
            elite_size = int(self.elite_ratio * self.population_size)
            elite_indices = sorted_indices[:elite_size]
            
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                if i in elite_indices:
                    # Perform Lévy flights on elite individuals
                    levy_step = self.levy_flight(self.dim)
                    mutated_individual = population[i] + levy_step
                    mutated_individual = np.clip(mutated_individual, self.lb, self.ub)
                else:
                    # Perform migration for non-elite individuals
                    random_idx = np.random.choice(elite_indices)
                    mutated_individual = population[i] + self.mutation_probability * (population[random_idx] - population[i])
                    mutated_individual = np.clip(mutated_individual, self.lb, self.ub)

                # Evaluate the mutated individual
                mutated_fitness = func(mutated_individual)
                num_evaluations += 1
                
                # Selection
                if mutated_fitness < fitness[i]:
                    population[i] = mutated_individual
                    fitness[i] = mutated_fitness
                    if mutated_fitness < best_fitness:
                        best_individual = mutated_individual
                        best_fitness = mutated_fitness

        return best_individual, best_fitness