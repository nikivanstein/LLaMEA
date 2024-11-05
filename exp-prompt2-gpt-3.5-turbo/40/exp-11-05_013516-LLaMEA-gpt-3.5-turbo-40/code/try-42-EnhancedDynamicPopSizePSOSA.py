import numpy as np

class EnhancedDynamicPopSizePSOSA:
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

        def simulated_annealing(current_pos, current_cost, best_pos, best_cost, T_max=1.0, T_min=0.001, alpha=0.9):
            T = T_max * (T_min / T_max) ** (iter / self.max_iter)
            candidate_pos = current_pos + np.random.uniform(-0.1, 0.1, size=self.dim)
            candidate_pos = np.clip(candidate_pos, -5.0, 5.0)

            candidate_cost = func(candidate_pos)
            if candidate_cost < current_cost or np.exp((current_cost - candidate_cost) / T) > np.random.rand():
                return candidate_pos, candidate_cost
            return current_pos, current_cost

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