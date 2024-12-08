import numpy as np
from scipy.optimize import differential_evolution

class CrowdSourcedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0])  # Search space between -5.0 and 5.0
        self.mean = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)  # Initialize mean
        self.covariance = np.eye(self.dim) * 1.0  # Initialize covariance matrix
        self.covariance_update_rate = 0.5  # Update covariance at 50% of the budget
        self.pso_particle_count = 10  # Initialize particle count for PSO

    def __call__(self, func):
        for i in range(int(self.budget * self.covariance_update_rate)):
            # Perform evolution strategy to update mean
            new_mean = self.mean + np.random.normal(0, 1.0, size=self.dim)
            new_mean = np.clip(new_mean, self.search_space[0], self.search_space[1])  # Clip values to search space

            # Perform genetic drift to update covariance
            new_covariance = self.covariance + np.random.normal(0, 0.1, size=(self.dim, self.dim))
            new_covariance = np.clip(new_covariance, 0, 1.0)  # Clip values to avoid negative covariance matrix

            # Evaluate function at new mean
            f_new = func(new_mean)

            # Update mean and covariance
            self.mean = new_mean
            self.covariance = new_covariance

            # Print current best solution
            print(f"Current best solution: x = {self.mean}, f(x) = {f_new}")

        # Perform PSO to improve convergence
        self.pso(self.mean, func)

        # Perform final optimization using differential evolution
        bounds = [(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=self.mean, seed=42)
        print(f"Final best solution: x = {res.x}, f(x) = {res.fun}")

    def pso(self, x, func):
        particles = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.pso_particle_count, self.dim))
        velocities = np.zeros((self.pso_particle_count, self.dim))
        best_particles = np.zeros((self.pso_particle_count, self.dim))
        best_fitness = np.inf

        for _ in range(10):
            for i in range(self.pso_particle_count):
                # Evaluate function at particle
                f_particle = func(particles[i])

                # Update velocities and particles
                velocities[i] = 0.5 * velocities[i] + 1.0 * np.random.normal(0, 0.1, size=self.dim)
                particles[i] = particles[i] + velocities[i]

                # Clip values to search space
                particles[i] = np.clip(particles[i], self.search_space[0], self.search_space[1])

                # Update best fitness and particles
                if f_particle < best_fitness:
                    best_fitness = f_particle
                    best_particles[i] = particles[i]

        # Update mean with best particles
        self.mean = np.mean(best_particles, axis=0)

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

crowd_sourced = CrowdSourcedMetaheuristic(budget=100, dim=2)
crowd_sourced(func)