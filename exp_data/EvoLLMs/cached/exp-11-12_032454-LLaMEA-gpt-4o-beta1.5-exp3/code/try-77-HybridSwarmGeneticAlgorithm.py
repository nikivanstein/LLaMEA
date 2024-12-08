import numpy as np

class HybridSwarmGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.c1 = 1.5  # personal learning coefficient
        self.c2 = 1.5  # global learning coefficient
        self.w = 0.7  # inertia weight
        self.mutation_rate = 0.1  # mutation probability
        self.crossover_rate = 0.8  # crossover probability
        
    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        pbest = np.copy(population)
        pbest_fitness = np.copy(fitness)
        
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Particle Swarm update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (pbest[i] - population[i]) + 
                                 self.c2 * r2 * (gbest - population[i]))
                candidate = population[i] + velocities[i]
                candidate = np.clip(candidate, self.lb, self.ub)
                
                # Evaluate candidate
                candidate_fitness = func(candidate)
                num_evaluations += 1
                
                # Personal best update
                if candidate_fitness < pbest_fitness[i]:
                    pbest[i] = candidate
                    pbest_fitness[i] = candidate_fitness
                    
                    # Global best update
                    if candidate_fitness < pbest_fitness[gbest_idx]:
                        gbest = candidate
                        gbest_idx = i
                
                # Apply Genetic Crossover
                if np.random.rand() < self.crossover_rate:
                    mate_idx = np.random.randint(self.population_size)
                    crossover_mask = np.random.rand(self.dim) < 0.5
                    candidate = np.where(crossover_mask, candidate, population[mate_idx])
                    
                # Apply Mutation
                if np.random.rand() < self.mutation_rate:
                    mutation_idx = np.random.randint(self.dim)
                    candidate[mutation_idx] = np.random.uniform(self.lb, self.ub)
                
                # Selection
                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
        
        return gbest, pbest_fitness[gbest_idx]