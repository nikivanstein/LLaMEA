import numpy as np

class HybridFireflyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.firefly_population_size = 10
        self.particle_swarm_size = 10

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // (self.firefly_population_size + self.particle_swarm_size)):
            # Firefly Algorithm
            firefly_population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.firefly_population_size)]
            firefly_fitness_values = [func(individual) for individual in firefly_population]
            
            for idx, fitness in enumerate(firefly_fitness_values):
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = firefly_population[idx]
            
            # Particle Swarm Optimization
            particle_swarm_population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.particle_swarm_size)]
            particle_swarm_fitness_values = [func(individual) for individual in particle_swarm_population]
            
            for idx, fitness in enumerate(particle_swarm_fitness_values):
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = particle_swarm_population[idx]
        
        return best_solution