class EnhancedCuckooSearch:
    def __init__(self, budget, dim, population_size=10, pa=0.25, alpha=0.01, elitism_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.pa = pa
        self.alpha = alpha
        self.elitism_rate = elitism_rate
        self.step_sizes = np.ones(self.population_size)

    def levy_flight(self, step_size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = step_size * u / abs(v) ** (1 / beta)
        return step

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        fitness = [func(x) for x in population]
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        for _ in range(self.budget):
            new_population = []
            for i, cuckoo in enumerate(population):
                step_size = self.step_sizes[i]
                cuckoo_new = cuckoo + self.levy_flight(step_size) * np.random.randn(self.dim)
                cuckoo_new = np.clip(cuckoo_new, -5.0, 5.0)

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

                    # Dynamic step size adaptation
                    if new_fitness < fitness[i]:
                        self.step_sizes[i] *= 1.1
                    else:
                        self.step_sizes[i] *= 0.9

            if np.random.rand() < self.elitism_rate:
                worst_idx = np.argmax(fitness)
                population[worst_idx] = best_solution
                fitness[worst_idx] = func(best_solution)

        return best_solution