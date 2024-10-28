import numpy as np

class ADELS:
    def __init__(self, budget, dim, n_particles=30, max_local_iter=10, mutation_rate=0.1, adapt_rate=0.3):
        self.budget = budget
        self.dim = dim
        self.n_particles = n_particles
        self.max_local_iter = max_local_iter
        self.mutation_rate = mutation_rate
        self.adapt_rate = adapt_rate

    def __call__(self, func):
        def local_search(x):
            best_x = x.copy()
            best_fitness = func(x)
            for _ in range(self.max_local_iter):
                new_x = x + np.random.uniform(-0.1, 0.1, size=self.dim)
                new_fitness = func(new_x)
                if new_fitness < best_fitness:
                    best_x = new_x
                    best_fitness = new_fitness
            return best_x

        swarm = np.random.uniform(-5.0, 5.0, size=(self.n_particles, self.dim))
        swarm_fitness = np.array([func(p) for p in swarm])
        best_idx = np.argmin(swarm_fitness)
        best_solution = swarm[best_idx]
        mutation_rates = np.full(self.n_particles, self.mutation_rate)

        for _ in range(self.budget // self.n_particles):
            for i in range(self.n_particles):
                new_position = swarm[i] + np.random.uniform(-1, 1, size=self.dim) * (best_solution - swarm[i])
                new_position = np.clip(new_position, -5.0, 5.0)

                if np.random.rand() < mutation_rates[i]:
                    new_position += np.random.normal(0, 0.5, size=self.dim)

                new_position = local_search(new_position)
                new_fitness = func(new_position)

                if new_fitness < swarm_fitness[i]:
                    swarm[i] = new_position
                    swarm_fitness[i] = new_fitness

                    if new_fitness < func(best_solution):
                        best_solution = new_position

                    mutation_rates[i] *= (1 - self.adapt_rate) + self.adapt_rate * np.random.rand()

        return best_solution