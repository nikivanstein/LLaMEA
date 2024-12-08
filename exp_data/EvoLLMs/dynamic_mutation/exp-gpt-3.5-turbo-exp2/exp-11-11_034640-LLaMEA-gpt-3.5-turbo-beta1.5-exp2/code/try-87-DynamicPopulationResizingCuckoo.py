import numpy as np

class DynamicPopulationResizingCuckoo:
    def __init__(self, budget, dim, initial_population_size=10, pa=0.25, alpha=0.01, elitism_rate=0.1, diversity_threshold=0.1, resize_rate=0.1, max_population_size=20, min_population_size=5):
        self.budget = budget
        self.dim = dim
        self.population_size = initial_population_size
        self.pa = pa
        self.alpha = alpha
        self.elitism_rate = elitism_rate
        self.diversity_threshold = diversity_threshold
        self.resize_rate = resize_rate
        self.max_population_size = max_population_size
        self.min_population_size = min_population_size

    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = u / abs(v) ** (1 / beta)
        return step

    def calculate_diversity(self, population):
        distances = np.linalg.norm(population[:, np.newaxis] - population, axis=2)
        np.fill_diagonal(distances, np.inf)
        return np.mean(np.min(distances, axis=1))

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        fitness = [func(x) for x in population]
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        fitness_improvement_rate = 0.0

        for _ in range(self.budget):
            new_population = []
            diversity = self.calculate_diversity(population)

            for i, cuckoo in enumerate(population):
                step_size = self.levy_flight() * (1 / (1 + fitness_improvement_rate)) * (1 + diversity * self.diversity_threshold)
                cuckoo_new = cuckoo + step_size * np.random.randn(self.dim)
                cuckoo_new = np.clip(cuckoo_new, -5.0, 5.0)

                if np.random.rand() > self.pa:
                    idx = np.random.randint(self.population_size)
                    cuckoo_new = cuckoo_new + self.alpha * (population[idx] - cuckoo_new)

                new_fitness = func(cuckoo_new)
                if new_fitness < fitness[i]:
                    population[i] = cuckoo_new
                    fitness[i] = new_fitness
                    fitness_improvement_rate = (fitness[i] - fitness_improvement_rate) / fitness[i]

                    if new_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = cuckoo_new

            if np.random.rand() < self.elitism_rate:
                worst_idx = np.argmax(fitness)
                population[worst_idx] = best_solution
                fitness[worst_idx] = func(best_solution)

            if np.random.rand() < self.resize_rate:
                if self.population_size < self.max_population_size:
                    self.population_size += 1
                elif self.population_size > self.min_population_size:
                    self.population_size -= 1

        return best_solution