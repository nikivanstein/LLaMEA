import numpy as np

class Improved_Vectorized_PSO_SA_Optimizer:
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

            new_values = np.array([func(p) for p in new_positions])
            better_indices = np.where(new_values < best_value)[0]
            best_value = np.min(new_values)
            gbest_position = new_positions[np.argmin(new_values)]

            update_indices = np.where(new_values < np.array([func(p) for p in pbest_positions]))[0]
            pbest_positions[update_indices] = new_positions[update_indices]

            particles, velocities = new_positions, new_velocities

            return particles, velocities, pbest_positions, gbest_position

        def sa_step(current_position, current_value, T, alpha=0.95):
            nonlocal best_value
            new_position = current_position + np.random.normal(0, T, size=(self.num_particles, self.dim))
            np.clip(new_position, -5.0, 5.0, out=new_position)
            new_values = np.array([func(p) for p in new_position])

            update_indices = np.where(new_values < current_value)[0]
            update_probabilities = np.exp((current_value - new_values) / T)
            accept_indices = np.where(np.random.rand(self.num_particles) < update_probabilities)[0]
            accept_indices = np.intersect1d(accept_indices, update_indices)

            current_position[accept_indices] = new_position[accept_indices]
            current_value = new_values[accept_indices]

            update_indices = np.where(new_values < best_value)[0]
            best_value = np.min(new_values[update_indices])

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