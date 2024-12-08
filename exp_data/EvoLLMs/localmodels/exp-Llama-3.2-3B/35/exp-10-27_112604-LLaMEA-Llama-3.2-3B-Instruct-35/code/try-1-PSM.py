import numpy as np

class PSM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.pbest = np.zeros((budget, dim))
        self.rbest = np.zeros((budget, dim))
        self.memory = {}
        self.probability = 0.35

    def __call__(self, func):
        for i in range(self.budget):
            # Evaluate the function at the current particles
            values = func(self.particles[i])
            # Update the personal best and global best
            self.pbest[i] = self.particles[i]
            self.rbest[i] = values
            # Check if the current best is better than the stored best
            if np.any(values < self.rbest, axis=1):
                self.memory[i] = values
                self.rbest[np.argmin(self.rbest, axis=1)] = values
            # Update the particles using the PSO update rule
            self.particles[i] = self.particles[i] + 0.5 * (self.pbest[i] - self.particles[i]) + 0.5 * (self.rbest - self.particles[i])
            # Line search using probability-based adaptation
            if np.random.rand() < self.probability:
                # Get the current particle and its corresponding best value
                particle = self.particles[i]
                best_value = self.rbest[i]
                # Calculate the step size for the line search
                step_size = np.random.uniform(0.1, 0.5)
                # Calculate the new particle position
                new_particle = particle + step_size * np.sign(particle)
                # Evaluate the new particle position
                new_value = func(new_particle)
                # Check if the new particle position is better than the current best
                if new_value < best_value:
                    self.particles[i] = new_particle
                    self.rbest[i] = new_value
        # Return the best solution found
        return self.rbest[np.argmin(self.rbest, axis=1)]

# Example usage
def func(x):
    return np.sum(x**2)

psm = PSM(budget=10, dim=2)
best_solution = psm(func)
print(best_solution)