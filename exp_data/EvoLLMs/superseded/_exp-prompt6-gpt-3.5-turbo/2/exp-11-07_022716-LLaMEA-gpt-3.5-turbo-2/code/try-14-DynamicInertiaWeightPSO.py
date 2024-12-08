import numpy as np

class DynamicInertiaWeightPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def inertia_weight_iter(curr_iter, max_iter):
            return 0.5 * (1 - curr_iter / max_iter)

        def rand_unit_vector(dim):
            vec = np.random.rand(dim)
            return vec / np.linalg.norm(vec)

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
            for i in range(len(population)):
                r1, r2 = rand_unit_vector(self.dim), rand_unit_vector(self.dim)
                velocity = inertia_weight * population[i] + 0.5 * r1 * (pbest[i] - population[i]) + 0.5 * r2 * (gbest - population[i])
                population[i] = np.clip(population[i] + velocity, -5.0, 5.0)
                fitness[i] = func(population[i])
                if fitness[i] < pbest_fitness[i]:
                    pbest[i] = population[i]
                    pbest_fitness[i] = fitness[i]
                    if fitness[i] < fitness[gbest_idx]:
                        gbest_idx = i
                        gbest = population[i]
                curr_iter += 1
                if curr_iter >= self.budget:
                    break

        return gbest