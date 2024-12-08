import numpy as np

class EnhancedCuckooSearchFastConvergence(EnhancedCuckooSearch):
    def __init__(self, budget, dim, population_size=10, pa=0.25, alpha=0.01, elitism_rate=0.1, step_size_factor=0.1):
        super().__init__(budget, dim, population_size, pa, alpha, elitism_rate)
        self.step_size_factor = step_size_factor

    def dynamic_step_size_adaptation(self, func, cuckoo, fitness):
        step_size = self.levy_flight()
        cuckoo_new = cuckoo + step_size * np.random.randn(self.dim)
        cuckoo_new = np.clip(cuckoo_new, -5.0, 5.0)

        new_fitness = func(cuckoo_new)
        
        if new_fitness < fitness:
            return cuckoo_new, new_fitness
        else:
            return cuckoo, fitness

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        fitness = [func(x) for x in population]
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        for _ in range(self.budget):
            new_population = []
            for i, cuckoo in enumerate(population):
                cuckoo, fitness[i] = self.dynamic_step_size_adaptation(func, cuckoo, fitness[i])

                if np.random.rand() > self.pa:
                    idx = np.random.randint(self.population_size)
                    cuckoo = cuckoo + self.alpha * (population[idx] - cuckoo)

                if fitness[i] < fitness[best_idx]:
                    best_idx = i
                    best_solution = cuckoo

            if np.random.rand() < self.elitism_rate:
                worst_idx = np.argmax(fitness)
                population[worst_idx] = best_solution
                fitness[worst_idx] = func(best_solution)

        return best_solution