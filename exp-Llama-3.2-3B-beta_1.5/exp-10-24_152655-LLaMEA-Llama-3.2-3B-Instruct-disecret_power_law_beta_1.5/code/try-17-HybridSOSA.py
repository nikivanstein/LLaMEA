import numpy as np
import random

class HybridSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.map_size = int(np.sqrt(dim))
        self.num_particles = 10
        self.w = 0.8
        self.c1 = 1.5
        self.c2 = 2.0

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Initialize a map
            map_ = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.map_size, self.map_size))

            # Initialize particles
            particles = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.num_particles, self.dim))

            # Evaluate particles
            f_particles = func(particles)

            # Update best solution
            f_evals = f_particles[0]
            x_best = particles[0]
            f_evals_best = f_evals

            # Update best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Update particles
            for i in range(self.num_particles):
                r1 = random.random()
                r2 = random.random()
                d1 = self.c1 * r1 * (f_particles[i] - f_evals_best) * (map_[i % self.map_size, :] - map_[np.argmin(f_particles, axis=0), :])
                d2 = self.c2 * r2 * (f_particles[i] - f_evals_best) * (particles[i] - x_best)
                particles[i] += d1 + d2

                # Check bounds
                particles[i] = np.clip(particles[i], self.bounds[:, 0], self.bounds[:, 1])

                # Evaluate new particle
                f_particles[i] = func(particles[i])

                # Update best solution if necessary
                if f_particles[i] < f_evals:
                    f_evals = f_particles[i]
                    x_best = particles[i]

            # Update map
            map_ = np.mean(particles, axis=0)

            # Update bounds
            self.bounds = np.array([np.min(particles, axis=0), np.max(particles, axis=0)])

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

hybrid_sosa = HybridSOSA(budget=10, dim=2)
x_opt = hybrid_sosa(func)
print(x_opt)