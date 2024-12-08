import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 2.0
        self.c2 = 2.0

    def __call__(self, func):
        def generate_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.dim, self.dim))

        particles = generate_particles()
        best_position = np.copy(particles[np.argmin([func(particle) for particle in particles])])
        velocities = np.zeros((self.dim, self.dim))

        w = self.w_max
        for _ in range(self.budget):
            for idx, particle in enumerate(particles):
                new_velocity = w * velocities[idx] + self.c1 * np.random.uniform() * (best_position - particle) + self.c2 * np.random.uniform() * (particles[np.argmin([func(p) for p in particles])] - particle)
                particles[idx] = np.clip(particle + new_velocity, -5.0, 5.0)

            if np.random.uniform() < 0.1:
                w = np.clip(w - 0.05, self.w_min, self.w_max)

            best_position = np.copy(particles[np.argmin([func(p) for p in particles])])

        return best_position