import numpy as np

class HybridSwarmAdaptiveOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.F = 0.5  # scaling factor for mutation
        self.CR = 0.9  # crossover probability

    def __call__(self, func):
        # Initialize population and velocities
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        pbest = np.copy(population)
        pbest_fitness = np.copy(fitness)
        gbest_idx = np.argmin(fitness)
        gbest = population[gbest_idx]
        gbest_fitness = fitness[gbest_idx]
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Update velocities and positions using PSO dynamics
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (pbest[i] - population[i]) + 
                                 self.c2 * r2 * (gbest - population[i]))
                population[i] = np.clip(population[i] + velocities[i], self.lb, self.ub)
                
                # Evaluate new candidate
                candidate_fitness = func(population[i])
                num_evaluations += 1
                
                # Update personal best
                if candidate_fitness < pbest_fitness[i]:
                    pbest[i] = population[i]
                    pbest_fitness[i] = candidate_fitness
                    
                    # Update global best
                    if candidate_fitness < gbest_fitness:
                        gbest = population[i]
                        gbest_fitness = candidate_fitness

                # Adaptive Differential Evolution Mutation and Crossover
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutated_vector = pbest[a] + self.F * (pbest[b] - pbest[c])
                mutated_vector = np.clip(mutated_vector, self.lb, self.ub)
                crossover_mask = np.random.rand(self.dim) < self.CR
                offspring = np.where(crossover_mask, mutated_vector, population[i])
                
                # Evaluate offspring
                offspring_fitness = func(offspring)
                num_evaluations += 1

                # Selection
                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness
                    if offspring_fitness < gbest_fitness:
                        gbest = offspring
                        gbest_fitness = offspring_fitness

        return gbest, gbest_fitness