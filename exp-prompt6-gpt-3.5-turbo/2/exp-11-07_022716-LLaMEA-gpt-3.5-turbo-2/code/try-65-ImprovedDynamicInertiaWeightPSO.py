import numpy as np

class ImprovedDynamicInertiaWeightPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def inertia_weight_iter(curr_iter, max_iter):
            return 0.5 * (1 - curr_iter / max_iter)

        def rand_unit_vectors(dim, num_vectors):
            vecs = np.random.rand(num_vectors, dim)
            norms = np.linalg.norm(vecs, axis=1)
            return vecs / norms[:, None]

        population = np.random.uniform(-5.0, 5.0, (10, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        gbest_idx = np.argmin(fitness)
        gbest = population[gbest_idx]
        pbest = np.copy(population)
        pbest_fitness = np.copy(fitness)

        curr_iter = 0
        while curr_iter < self.budget:
            inertia_weight = inertia_weight_iter(curr_iter, self.budget)
            r1, r2 = rand_unit_vectors(self.dim, len(population)), rand_unit_vectors(self.dim, len(population))
            velocities = inertia_weight * population + 0.5 * r1 * (pbest - population)[:, None] + 0.5 * r2 * (gbest - population)[:, None]
            population = np.clip(population + velocities, -5.0, 5.0)
            fitness = np.apply_along_axis(func, 1, population)
            updates = fitness < pbest_fitness
            pbest[updates] = population[updates]
            pbest_fitness[updates] = fitness[updates]
            gbest_idx = np.argmin(fitness)
            gbest = population[gbest_idx]
            curr_iter += len(population)

        return gbest