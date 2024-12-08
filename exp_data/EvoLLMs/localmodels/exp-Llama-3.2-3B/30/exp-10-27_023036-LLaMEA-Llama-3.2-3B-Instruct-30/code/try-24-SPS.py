import numpy as np

class SPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_particles = np.zeros((self.population_size, self.dim))
        self.pbest = np.zeros((self.population_size, self.dim))
        self.rbest = np.zeros((self.population_size, self.dim))
        self.candidates = []

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the objective function
            values = func(self.particles)
            # Update the best particles
            for i in range(self.population_size):
                if values[i] < self.best_particles[i, 0]:
                    self.best_particles[i] = values[i]
                    self.pbest[i] = values[i]
                    self.rbest[i] = values[i]
            # Update the swarm
            for i in range(self.population_size):
                if np.random.rand() < 0.3:
                    self.particles[i] += np.random.uniform(-1, 1, self.dim)
                if np.random.rand() < 0.3:
                    self.particles[i] += np.random.uniform(0, 1, self.dim)
                if np.random.rand() < 0.3:
                    self.particles[i] += self.pbest[i] - self.particles[i]
                if np.random.rand() < 0.3:
                    self.particles[i] += self.rbest[i] - self.particles[i]
            # Evaluate the objective function again
            values = func(self.particles)
            # Update the candidates
            for i in range(self.population_size):
                if values[i] > self.rbest[i, 0]:
                    self.rbest[i] = values[i]
                    self.candidates.append([self.particles[i], values[i]])
            # Update the swarm
            self.particles = np.array(self.candidates)
            self.candidates = []
            # Sort the particles
            self.particles = self.particles[np.argsort(self.rbest, axis=0)]
            # Update the best particles
            self.best_particles = self.particles[:, 0]
        # Return the best particle
        return self.particles[:, 0]

# Example usage
def func(x):
    return np.sum(x**2)

sp = SPS(budget=100, dim=10)
best_particle = sp(func)
print(best_particle)