import numpy as np

class ImprovedDynamicInertiaWeightPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def inertia_weight_iter(curr_iter, max_iter):
            return 0.5 * (1 - curr_iter / max_iter)

        def generate_population(pop_size):
            return np.random.uniform(-5.0, 5.0, (pop_size, self.dim))

        population = generate_population(10)
        fitness = np.apply_along_axis(func, 1, population)
        gbest_idx = np.argmin(fitness)
        gbest = population[gbest_idx]
        pbest = np.copy(population)
        pbest_fitness = np.copy(fitness)

        curr_iter = 0
        while curr_iter < self.budget:
            inertia_weight = inertia_weight_iter(curr_iter, self.budget)
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocity = inertia_weight * population + 0.5 * r1 * (pbest - population) + 0.5 * r2 * (gbest - population)
            population = np.clip(population + velocity, -5.0, 5.0)
            fitness = np.apply_along_axis(func, 1, population)
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = population[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]
            gbest_idx = np.argmin(fitness)
            gbest = population[gbest_idx]
            curr_iter += 1
            if curr_iter >= self.budget:
                break

        return gbest