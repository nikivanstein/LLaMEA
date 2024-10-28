import numpy as np

class MultiSwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.particles = []
        self.best_particles = []
        self.crossover_probability = 0.5
        self.mutation_probability = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            for _ in range(self.swarm_size):
                particle = np.random.uniform(-5.0, 5.0, self.dim)
                self.particles.append(particle)

            for i in range(self.swarm_size):
                particle = self.particles[i]
                # Evaluate the function
                fitness = func(particle)

                # Update the best particle
                if fitness < func(self.best_particles[i]):
                    self.best_particles[i] = particle

                # Update the particle
                if np.random.rand() < self.crossover_probability:
                    # Crossover
                    r1, r2 = np.random.permutation(self.dim)
                    particle[r1], particle[r2] = particle[r2], particle[r1]
                if np.random.rand() < self.mutation_probability:
                    # Mutation
                    r = np.random.randint(0, self.dim)
                    particle[r] += np.random.uniform(-0.1, 0.1)

            # Update the best particles
            for i in range(self.swarm_size):
                self.best_particles[i] = self.particles[i]

        # Return the best particle
        return min(self.best_particles, key=func)

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = MultiSwarmDE(budget, dim)
best_solution = optimizer(func)
print("Best solution:", best_solution)