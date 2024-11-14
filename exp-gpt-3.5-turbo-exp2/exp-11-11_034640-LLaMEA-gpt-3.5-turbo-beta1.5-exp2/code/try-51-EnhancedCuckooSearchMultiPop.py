import numpy as np

class EnhancedCuckooSearchMultiPop:
    def __init__(self, budget, dim, population_size=10, pa=0.25, alpha=0.01, elitism_rate=0.1, num_pops=3, share_interval=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.pa = pa
        self.alpha = alpha
        self.elitism_rate = elitism_rate
        self.num_pops = num_pops
        self.share_interval = share_interval

    def __call__(self, func):
        populations = [np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim)) for _ in range(self.num_pops)]
        fitness = [[func(x) for x in pop] for pop in populations]
        best_solutions = [pop[np.argmin(fit)] for pop, fit in zip(populations, fitness)]
        
        for _ in range(self.budget):
            for i, (population, pop_fit, best_sol) in enumerate(zip(populations, fitness, best_solutions)):
                new_population = []
                for j, cuckoo in enumerate(population):
                    step_size = self.levy_flight()
                    cuckoo_new = cuckoo + step_size * np.random.randn(self.dim)
                    cuckoo_new = np.clip(cuckoo_new, -5.0, 5.0)

                    if np.random.rand() > self.pa:
                        idx = np.random.randint(self.population_size)
                        cuckoo_new = cuckoo_new + self.alpha * (population[idx] - cuckoo_new)

                    new_fitness = func(cuckoo_new)
                    if new_fitness < pop_fit[j]:
                        population[j] = cuckoo_new
                        pop_fit[j] = new_fitness

                        if new_fitness < func(best_sol):
                            best_solutions[i] = cuckoo_new

                if np.random.rand() < self.elitism_rate:
                    worst_idx = np.argmax(pop_fit)
                    population[worst_idx] = best_sol
                    pop_fit[worst_idx] = func(best_sol)

            if _ % self.share_interval == 0:
                best_global_sol = min(best_solutions, key=lambda x: func(x))
                for i in range(self.num_pops):
                    best_solutions[i] = best_global_sol

        return min(best_solutions, key=lambda x: func(x))