import random
import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = []
        self.best_particle = None
        self.best_func_value = float('inf')

    def __call__(self, func):
        if self.budget <= 0:
            return self.best_particle

        # Initialize particles
        for _ in range(self.population_size):
            particle = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            self.particles.append(particle)

        for _ in range(self.budget):
            # Evaluate particles
            values = [func(p) for p in self.particles]
            best_particle_index = np.argmin(values)

            # Update best particle
            if values[best_particle_index] < self.best_func_value:
                self.best_particle = self.particles[best_particle_index]
                self.best_func_value = values[best_particle_index]

            # Update particles
            for i, particle in enumerate(self.particles):
                # Crossover
                if random.random() < 0.3:
                    crossover_point = random.randint(0, self.dim - 1)
                    particle[crossover_point:] = self.particles[best_particle_index][crossover_point:]

                # Mutation
                if random.random() < 0.3:
                    mutation_point = random.randint(0, self.dim - 1)
                    particle[mutation_point] += random.uniform(-1.0, 1.0)

                # Update velocity
                velocity = [0.0] * self.dim
                velocity[best_particle_index] = values[best_particle_index] - func(particle)
                self.particles[i] = [p + v for p, v in zip(particle, velocity)]

        return self.best_particle

# Example usage
def func(x):
    return sum([i**2 for i in x])

budget = 100
dim = 10
optimizer = AdaptivePSO(budget, dim)
best_particle = optimizer(func)
print(best_particle)