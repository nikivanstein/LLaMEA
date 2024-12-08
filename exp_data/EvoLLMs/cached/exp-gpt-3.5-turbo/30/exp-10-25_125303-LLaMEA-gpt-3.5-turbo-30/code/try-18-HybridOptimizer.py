import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim, swarm_size=20, alpha=0.9, initial_temp=10.0, final_temp=0.1):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.initial_temp = initial_temp
        self.final_temp = final_temp

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        def hybrid_step(swarm, best_particle, temperature):
            new_swarm = []
            for particle in swarm:
                velocity = np.random.uniform() * (best_particle - particle)
                new_particle = particle + velocity
                new_particle = new_particle + np.random.uniform(-1, 1) * temperature
                new_swarm.append(new_particle)
            return np.array(new_swarm)

        swarm = initialize_population()
        best_particle = swarm[np.argmin([func(p) for p in swarm])
        temperature = self.initial_temp
        remaining_budget = self.budget - self.swarm_size

        while remaining_budget > 0 and temperature > self.final_temp:
            new_swarm = hybrid_step(swarm, best_particle, temperature)
            for idx, particle in enumerate(new_swarm):
                new_fitness = func(particle)
                if new_fitness < func(swarm[idx]):
                    swarm[idx] = particle
                    if new_fitness < func(best_particle):
                        best_particle = particle
                remaining_budget -= 1
                if remaining_budget <= 0 or temperature <= self.final_temp:
                    break
            temperature *= self.alpha

        return best_particle