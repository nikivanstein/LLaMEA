import numpy as np
import random
import operator

class HybridAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.swarm_particles = []
        self adaptive_model = {}

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Initialize a new particle
            particle = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))
            self.swarm_particles.append(particle)

            # Evaluate the particle
            f_particle = func(particle)
            self.f_evals += 1

            # Update the best solution
            f_evals = f_particle
            x_best = particle
            f_evals_best = f_evals

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Adaptive dynamic programming
            for i, particle in enumerate(self.swarm_particles):
                # Compute the fitness value
                f_particle = func(particle)

                # Update the adaptive model
                if i not in self.adaptive_model:
                    self.adaptive_model[i] = {}
                if particle not in self.adaptive_model[i]:
                    self.adaptive_model[i][particle] = 0
                self.adaptive_model[i][particle] += f_particle

                # Compute the weighted average
                weights = [f_particle / self.adaptive_model[i][particle] for particle in self.swarm_particles]
                self.adaptive_model[i]['avg'] = np.sum([weights[i] * particle for particle in self.swarm_particles]) / np.sum(weights)

            # Select the best particle
            best_particle = self.swarm_particles[np.argmin([func(particle) for particle in self.swarm_particles])]

            # Update the bounds
            self.bounds = np.array([np.min([func(particle) for particle in self.swarm_particles]), np.max([func(particle) for particle in self.swarm_particles])])

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

hybrid = HybridAlgorithm(budget=10, dim=2)
x_opt = hybrid(func)
print(x_opt)