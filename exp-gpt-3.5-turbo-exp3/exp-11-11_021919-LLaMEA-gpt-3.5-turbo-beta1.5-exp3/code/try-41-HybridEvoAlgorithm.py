import numpy as np

class HybridEvoAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iterations = budget // self.population_size
        self.w = 0.5
        self.c1 = 2.0
        self.c2 = 2.0
        self.F_min, self.F_max = 0.2, 0.8
        self.CR_min, self.CR_max = 0.2, 0.8
        
    def generate_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        
    def differential_evolution(self, population, func):
        # Implementation of Differential Evolution
        # Insert your DE algorithm here
        return population
        
    def particle_swarm_optimization(self, population, func):
        # Implementation of Particle Swarm Optimization
        # Insert your PSO algorithm here
        return population
        
    def __call__(self, func):
        population = self.generate_population()
        for _ in range(self.max_iterations):
            population = self.differential_evolution(population, func)
            population = self.particle_swarm_optimization(population, func)
        return np.min([func(individual) for individual in population])