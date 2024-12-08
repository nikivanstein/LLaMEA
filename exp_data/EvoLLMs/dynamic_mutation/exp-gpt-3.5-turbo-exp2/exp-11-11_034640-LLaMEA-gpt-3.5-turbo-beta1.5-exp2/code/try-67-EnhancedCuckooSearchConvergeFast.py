import numpy as np

class EnhancedCuckooSearchConvergeFast:
    def __init__(self, budget, dim, population_size=10, pa=0.25, alpha=0.01, elitism_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.pa = pa
        self.alpha = alpha
        self.elitism_rate = elitism_rate

    def levy_flight(self, fitness):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=len(fitness))
        v = np.random.normal(0, 1, size=len(fitness))
        steps = u / abs(v) ** (1 / beta)
        return steps

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        fitness = [func(x) for x in population]
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        for _ in range(self.budget):
            steps = self.levy_flight(fitness)
            new_population = population + steps[:, np.newaxis] * np.random.randn(self.population_size, self.dim)
            new_population = np.clip(new_population, -5.0, 5.0)

            for i, cuckoo_new in enumerate(new_population):
                if np.random.rand() > self.pa:
                    idx = np.random.randint(self.population_size)
                    cuckoo_new = cuckoo_new + self.alpha * (population[idx] - cuckoo_new)

                new_fitness = func(cuckoo_new)
                if new_fitness < fitness[i]:
                    population[i] = cuckoo_new
                    fitness[i] = new_fitness

                    if new_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = cuckoo_new

            # Introduce dynamic step size adaptation based on individual fitness
            step_fitness_ratio = np.array([fitness[i] / max(fitness) for i in range(self.population_size)])
            step_sizes = self.levy_flight(step_fitness_ratio)
            population += step_sizes[:, np.newaxis] * np.random.randn(self.population_size, self.dim)
            population = np.clip(population, -5.0, 5.0)

            if np.random.rand() < self.elitism_rate:
                worst_idx = np.argmax(fitness)
                population[worst_idx] = best_solution
                fitness[worst_idx] = func(best_solution)

        return best_solution