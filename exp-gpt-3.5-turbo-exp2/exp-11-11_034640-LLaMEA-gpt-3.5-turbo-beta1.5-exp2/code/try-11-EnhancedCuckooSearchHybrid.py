import numpy as np

class EnhancedCuckooSearchHybrid:
    def __init__(self, budget, dim, population_size=10, pa=0.25, alpha=0.01, elitism_rate=0.1, local_search_rate=0.3):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.pa = pa
        self.alpha = alpha
        self.elitism_rate = elitism_rate
        self.local_search_rate = local_search_rate

    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = u / abs(v) ** (1 / beta)
        return step

    def gradient_descent(self, x, func):
        lr = 0.1
        for _ in range(10):
            grad = np.gradient(func(x))
            x = x - lr * grad
            x = np.clip(x, -5.0, 5.0)
        return x

    def particle_swarm_optimization(self, x, func):
        inertia_weight = 0.5
        cognitive_weight = 1.0
        social_weight = 2.0
        personal_best = x
        global_best = x

        for _ in range(10):
            r1, r2 = np.random.random(), np.random.random()
            personal_best = inertia_weight * personal_best + cognitive_weight * r1 * (personal_best - x) + social_weight * r2 * (global_best - x)
            x = x + personal_best
            x = np.clip(x, -5.0, 5.0)
        return x

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        fitness = [func(x) for x in population]
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        for _ in range(self.budget):
            new_population = []
            for i, cuckoo in enumerate(population):
                step_size = self.levy_flight()
                cuckoo_new = cuckoo + step_size * np.random.randn(self.dim)
                cuckoo_new = np.clip(cuckoo_new, -5.0, 5.0)

                if np.random.rand() > self.pa:
                    idx = np.random.randint(self.population_size)
                    cuckoo_new = cuckoo_new + self.alpha * (population[idx] - cuckoo_new)

                # Introduce diverse local search mechanism
                if np.random.rand() < self.local_search_rate:
                    if np.random.rand() < 0.5:
                        cuckoo_new = self.gradient_descent(cuckoo_new, func)
                    else:
                        cuckoo_new = self.particle_swarm_optimization(cuckoo_new, func)

                new_fitness = func(cuckoo_new)
                if new_fitness < fitness[i]:
                    population[i] = cuckoo_new
                    fitness[i] = new_fitness

                    if new_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = cuckoo_new

            if np.random.rand() < self.elitism_rate:
                worst_idx = np.argmax(fitness)
                population[worst_idx] = best_solution
                fitness[worst_idx] = func(best_solution)

        return best_solution