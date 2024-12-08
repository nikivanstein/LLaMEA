import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim, swarm_size=20, alpha=0.9, initial_temp=10.0, final_temp=0.1, mutation_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        def pso_step(swarm, best_particle):
            new_swarm = []
            for particle in swarm:
                velocity = np.random.uniform() * (best_particle - particle)
                new_particle = particle + velocity
                new_swarm.append(new_particle)
            return np.array(new_swarm)

        def sa_step(current, best, temp):
            candidate = current + np.random.uniform(-1, 1) * temp
            candidate_fitness = func(candidate)
            current_fitness = func(current)
            if candidate_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - candidate_fitness) / temp):
                return candidate
            return current

        def mutate_particle(particle):
            mutated_particle = particle.copy()
            for i in range(len(particle)):
                if np.random.rand() < self.mutation_prob:
                    mutated_particle[i] += np.random.normal(0, 1)
            return mutated_particle

        swarm = initialize_population()
        best_particle = swarm[np.argmin([func(p) for p in swarm])]
        temperature = self.initial_temp
        remaining_budget = self.budget - self.swarm_size

        while remaining_budget > 0 and temperature > self.final_temp:
            new_swarm = pso_step(swarm, best_particle)
            for idx, particle in enumerate(new_swarm):
                new_particle = sa_step(particle, best_particle, temperature)
                new_particle = mutate_particle(new_particle)
                new_fitness = func(new_particle)
                if new_fitness < func(swarm[idx]):
                    swarm[idx] = new_particle
                    if new_fitness < func(best_particle):
                        best_particle = new_particle
                remaining_budget -= 1
                if remaining_budget <= 0 or temperature <= self.final_temp:
                    break
            temperature *= self.alpha

        return best_particle

# Example usage:
# optimizer = EnhancedHybridOptimizer(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function