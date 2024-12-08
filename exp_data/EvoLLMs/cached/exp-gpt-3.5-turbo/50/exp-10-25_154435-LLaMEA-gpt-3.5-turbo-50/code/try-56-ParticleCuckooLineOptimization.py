import numpy as np

class ParticleCuckooLineOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.dim,))

        def levy_flight(step_size):
            beta = 1.5
            sigma = (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma, size=(self.dim,))
            v = np.random.normal(0, 1, size=(self.dim,))
            step = u / (np.abs(v) ** (1 / beta))
            return step_size * step

        best_solution = initialize_particles()
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            for _ in range(5):
                particles = [initialize_particles() for _ in range(10)]
                for particle in particles:
                    step_size = np.random.uniform(0.1, 1.0)
                    new_particle = particle + levy_flight(step_size)
                    new_fitness = func(new_particle)
                    if new_fitness < best_fitness:
                        best_solution = new_particle
                        best_fitness = new_fitness

                # Cuckoo Search
                cuckoo = particles[np.argmax([func(p) for p in particles])]
                new_solution = best_solution + 0.1 * (cuckoo - best_solution)
                new_fitness = func(new_solution)
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness

                    # Line Refinement
                    if np.random.uniform() < 0.35:
                        line_direction = np.random.uniform(-1, 1, size=(self.dim,))
                        line_direction /= np.linalg.norm(line_direction)
                        line_length = np.random.uniform(0.1, 1.0)
                        line_point = best_solution + line_length * line_direction
                        line_fitness = func(line_point)
                        if line_fitness < best_fitness:
                            best_solution = line_point
                            best_fitness = line_fitness

        return best_solution