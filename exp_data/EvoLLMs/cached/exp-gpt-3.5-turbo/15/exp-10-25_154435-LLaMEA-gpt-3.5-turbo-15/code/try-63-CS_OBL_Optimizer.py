import numpy as np

class CS_OBL_Optimizer:
    def __init__(self, budget, dim, population_size=50, pa=0.25, alpha=1.5, beta=1.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.pa = pa
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        def levy_flight(dim):
            beta = 1.5
            sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma, dim)
            v = np.random.normal(0, 1, dim)
            step = u / np.abs(v) ** (1 / beta)
            return step

        def clip_bounds(population):
            return np.clip(population, -5.0, 5.0)

        def generate_opposite(individual):
            return -individual

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        for _ in range(self.budget - self.population_size):
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                if np.random.rand() > self.pa:
                    step = levy_flight(self.dim)
                    new_individual = population[i] + step * self.alpha * (population[i] - best_solution)
                    new_individual = clip_bounds(new_individual)
                    if func(new_individual) < fitness[i]:
                        new_population[i] = new_individual
                    else:
                        new_population[i] = population[i]
                else:
                    new_population[i] = generate_opposite(population[i])
                    
            population = new_population
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

        return best_solution