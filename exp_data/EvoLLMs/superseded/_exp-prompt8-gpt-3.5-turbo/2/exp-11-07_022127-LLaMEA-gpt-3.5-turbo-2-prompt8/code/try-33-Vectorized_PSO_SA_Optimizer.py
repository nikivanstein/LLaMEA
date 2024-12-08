import numpy as np

class Vectorized_PSO_SA_Optimizer:
    def __init__(self, budget, dim, num_particles=30, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter

    def __call__(self, func):
        def pso_step(particles, velocities, pbest_positions, gbest_position, w=0.5, c1=1.5, c2=1.5):
            nonlocal best_value
            random_values = np.random.rand(2, len(particles))
            new_velocities = w * velocities + c1 * random_values[0] * (pbest_positions - particles) + c2 * random_values[1] * (gbest_position - particles)
            new_positions = particles + new_velocities

            new_values = func(new_positions)
            improve = new_values < best_value
            best_value = np.where(improve, new_values, best_value)
            gbest_position = np.where(improve, new_positions, gbest_position)

            update_pbest = new_values < func(pbest_positions)
            pbest_positions = np.where(update_pbest[:, np.newaxis], new_positions, pbest_positions)

            particles, velocities = new_positions, new_velocities

            return particles, velocities, pbest_positions, gbest_position
            
        def sa_step(current_position, current_value, T, alpha=0.95):
            nonlocal best_value
            new_positions = current_position + np.random.normal(0, T, size=(self.num_particles, self.dim))
            new_positions = np.clip(new_positions, -5.0, 5.0)
            new_values = func(new_positions)

            update = new_values < current_value | np.random.rand(self.num_particles) < np.exp((current_value - new_values) / T)
            current_position = np.where(update[:, np.newaxis], new_positions, current_position)
            current_value = np.where(update, new_values, current_value)

            best_value = np.minimum(best_value, new_values)

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
            T *= 0.95  # Cooling

        return best_value