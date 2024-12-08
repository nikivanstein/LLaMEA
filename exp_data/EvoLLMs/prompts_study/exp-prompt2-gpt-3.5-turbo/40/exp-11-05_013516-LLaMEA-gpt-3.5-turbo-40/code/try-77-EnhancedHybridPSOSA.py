import numpy as np

class EnhancedHybridPSOSA:
    def __init__(self, budget, dim, num_particles=30, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(self.lower_bound, self.upper_bound, size=(self.num_particles, self.dim))

        def update_position(position, velocity):
            new_position = position + velocity
            return np.clip(new_position, self.lower_bound, self.upper_bound)

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
                particles[i], pbest_costs[i] = simulated_annealing(particles[i], pbest_costs[i], pbest[i], pbest_costs[i])

                if pbest_costs[i] < func(pbest[i]):
                    pbest[i] = particles[i]

            if pbest_costs.min() < gbest_cost:
                gbest = pbest[pbest_costs.argmin()]
                gbest_cost = pbest_costs.min()

        return gbest