import numpy as np

class EQPSO:
    def __init__(self, budget, dim, num_particles=30, inertia_weight=0.9, cognitive_weight=2.0, social_weight=2.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        def update_position(particles, velocities):
            return np.clip(particles + velocities, -5.0, 5.0)

        def evaluate_fitness(particles):
            return np.array([func(p) for p in particles])

        particles = initialize_particles()
        best_global_position = particles[np.argmin(evaluate_fitness(particles))]
        velocities = np.zeros_like(particles)
        cognitive_weight = 2.0
        social_weight = 2.0

        for _ in range(self.budget):
            for i in range(self.num_particles):
                rand1 = np.random.rand(self.dim)
                rand2 = np.random.rand(self.dim)
                cognitive_weight = 2.0 / (1 + np.exp(-0.1 * (func(particles[i]) - func(best_global_position))))
                social_weight = 2.0 / (1 + np.exp(-0.1 * (func(particles[i]) - func(best_global_position)))

                velocities[i] = self.inertia_weight * velocities[i] + \
                                cognitive_weight * rand1 * (best_global_position - particles[i]) + \
                                social_weight * rand2 * (best_global_position - particles[i])

                particles[i] = update_position(particles[i], velocities[i])

            fitness_values = evaluate_fitness(particles)
            best_particle_index = np.argmin(fitness_values)
            if fitness_values[best_particle_index] < func(best_global_position):
                best_global_position = particles[best_particle_index]

        return best_global_position