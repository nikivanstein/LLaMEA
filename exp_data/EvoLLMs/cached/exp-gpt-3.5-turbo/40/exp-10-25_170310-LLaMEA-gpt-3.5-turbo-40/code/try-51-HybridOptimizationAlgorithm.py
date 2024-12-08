import numpy as np

class HybridOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.inertia_weight = 0.5
        self.c1 = 2.05
        self.c2 = 2.05
        self.initial_temperature = 100
        self.min_temperature = 0.1

    def __call__(self, func):
        temperature = self.initial_temperature
        for _ in range(self.budget):
            for i in range(self.budget):
                particle = self.population[i]
                
                # Particle Swarm Optimization
                velocity = np.random.uniform(-1, 1, self.dim)
                personal_best = particle.copy()
                global_best = self.population[np.argmin([func(x) for x in self.population])]
                inertia_weight = self.inertia_weight
                c1, c2 = self.c1, self.c2
                velocity = inertia_weight * velocity + c1 * np.random.rand() * (personal_best - particle) + c2 * np.random.rand() * (global_best - particle)
                particle += velocity
                particle = np.clip(particle, -5.0, 5.0)
                
                # Simulated Annealing
                new_particle = particle + np.random.normal(0, temperature, self.dim)
                new_particle = np.clip(new_particle, -5.0, 5.0)
                energy_diff = func(new_particle) - func(particle)
                if energy_diff < 0 or np.random.rand() < np.exp(-energy_diff / temperature):
                    particle = new_particle
                
                if func(particle) < func(self.population[i]):
                    self.population[i] = particle
            
            temperature = max(temperature * 0.99, self.min_temperature)
        
        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]

        return best_solution