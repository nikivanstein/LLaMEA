import numpy as np

class EnhancedHybridPSOSAImproved:
    def __init__(self, budget, dim, num_particles=30, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        def update_velocity(position, velocity, pbest, gbest, w_min=0.4, w_max=0.9, c1=1.5, c2=2.0):
            w = w_min + (w_max - w_min) * (self.max_iter - iter) / self.max_iter
            r1, r2 = np.random.rand(2, self.dim)
            new_velocity = w * velocity + c1 * r1 * (pbest - position) + c2 * r2 * (gbest - position)
            return np.clip(new_velocity, -0.2, 0.2)

        def update_position(position, velocity):
            new_position = position + velocity
            return np.clip(new_position, -5.0, 5.0)

        def dynamic_mutation(position, mutation_rate=0.1):
            mutated_position = position + mutation_rate * np.random.randn(self.dim)
            return np.clip(mutated_position, -5.0, 5.0)

        particles = initialize_particles()
        velocities = np.zeros((self.num_particles, self.dim))
        pbest = particles.copy()
        pbest_costs = np.array([func(p) for p in particles])
        gbest = pbest[pbest_costs.argmin()]
        gbest_cost = pbest_costs.min()

        for iter in range(self.max_iter):
            for i in range(self.num_particles):
                velocities[i] = update_velocity(particles[i], velocities[i], pbest[i], gbest)
                particles[i] = update_position(particles[i], velocities[i])
                
                if np.random.rand() < 0.2:  # 20% chance of applying the mutation
                    particles[i] = dynamic_mutation(particles[i])

                particles[i], pbest_costs[i] = simulated_annealing(particles[i], pbest_costs[i], pbest[i], pbest_costs[i])

                if pbest_costs[i] < func(pbest[i]):
                    pbest[i] = particles[i]

            if pbest_costs.min() < gbest_cost:
                gbest = pbest[pbest_costs.argmin()]
                gbest_cost = pbest_costs.min()

        return gbest