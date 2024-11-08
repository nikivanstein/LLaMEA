import numpy as np

class Improved_PSO_SA_Optimizer:
    def __init__(self, budget, dim, num_particles=30, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter

    def __call__(self, func):
        def pso_step(particles, velocities, pbest_positions, gbest_position, w=0.5, c1=1.5, c2=1.5):
            nonlocal best_value
            for i in range(len(particles)):
                particle, velocity, pbest_position = particles[i], velocities[i], pbest_positions[i]

                r1, r2 = np.random.rand(), np.random.rand()
                new_velocity = w * velocity + c1 * r1 * (pbest_position - particle) + c2 * r2 * (gbest_position - particle)
                new_position = particle + new_velocity

                new_value = func(new_position)
                if new_value < best_value:
                    best_value, gbest_position = new_value, new_position

                if new_value < func(pbest_position):
                    pbest_positions[i] = new_position

                particles[i], velocities[i] = new_position, new_velocity

            return particles, velocities, pbest_positions, gbest_position

        def sa_step(current_position, current_value, T, alpha=0.95):
            nonlocal best_value
            new_position = np.clip(current_position + np.random.normal(0, T, size=self.dim), -5.0, 5.0)
            new_value = func(new_position)

            if new_value < current_value or np.random.rand() < np.exp((current_value - new_value) / T):
                current_position, current_value = new_position, new_value

            if new_value < best_value:
                best_value = new_value

            return current_position, current_value

        particles = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        velocities = np.zeros_like(particles)
        pbest_positions = particles.copy()
        gbest_position = particles[np.argmin([func(p) for p in particles])]
        best_value = func(gbest_position)

        T = 1.0
        current_position = np.mean(particles, axis=0)
        current_value = func(current_position)

        for _ in range(self.max_iter):
            particles, velocities, pbest_positions, gbest_position = pso_step(particles, velocities, pbest_positions, gbest_position)
            current_position, current_value = sa_step(current_position, current_value, T)
            T *= 0.95

        return best_value