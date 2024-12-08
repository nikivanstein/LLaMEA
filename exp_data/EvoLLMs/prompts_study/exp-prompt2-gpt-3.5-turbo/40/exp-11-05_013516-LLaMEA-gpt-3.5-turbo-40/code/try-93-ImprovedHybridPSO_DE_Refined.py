import numpy as np

class ImprovedHybridPSO_DE_Refined:
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

        def differential_evolution(current_pos, F=0.5, CR=0.9):
            candidates = np.random.permutation(particles)
            mutant = current_pos + F * (candidates[0] - candidates[1])
            trial = np.where(np.random.rand(self.dim) < CR, mutant, current_pos)
            trial = np.clip(trial, -5.0, 5.0)
            
            trial_cost = func(trial)
            if trial_cost < func(current_pos):
                return trial, trial_cost
            return current_pos, func(current_pos)

        def local_search(position, epsilon=0.01):
            new_position = position + epsilon * np.random.randn(self.dim)
            new_position = np.clip(new_position, -5.0, 5.0)
            return new_position

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
                particles[i], pbest_costs[i] = differential_evolution(particles[i])

                if pbest_costs[i] < func(pbest[i]):
                    pbest[i] = particles[i]

                # Integrate local search
                new_particle = local_search(particles[i])
                new_cost = func(new_particle)
                if new_cost < pbest_costs[i]:
                    particles[i] = new_particle
                    pbest_costs[i] = new_cost

            if pbest_costs.min() < gbest_cost:
                gbest = pbest[pbest_costs.argmin()]
                gbest_cost = pbest_costs.min()

        return gbest