import numpy as np

class EnhancedCuckooSearch:
    def __init__(self, budget, dim, population_size=10, pa=0.25, alpha=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.pa = pa
        self.alpha = alpha

    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u =  np.random.normal(0, sigma)
        v =  np.random.normal(0, 1)
        step =  u / abs(v) ** (1 / beta)
        return step

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
                
                new_fitness = func(cuckoo_new)
                if new_fitness < fitness[i]:
                    population[i] = cuckoo_new
                    fitness[i] = new_fitness
                    
                    if new_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = cuckoo_new
                        
            # Dynamic population size adaptation
            if np.random.rand() < 0.1:  # Adjust the probability for adaptation
                new_size = int(np.clip(np.round(self.population_size * np.random.normal(1, 0.1)), 2, 100))
                if new_size != self.population_size:
                    if new_size > self.population_size:
                        population = np.vstack([population, np.random.uniform(-5.0, 5.0, size=(new_size - self.population_size, self.dim))])
                        fitness.extend([func(x) for x in population[self.population_size:]])
                    else:
                        indices = np.random.choice(np.arange(self.population_size), new_size, replace=False)
                        population = population[indices]
                        fitness = [fitness[j] for j in indices]
                    self.population_size = new_size
        
        return best_solution