import numpy as np

class Fireworks_Optimizer:
    def __init__(self, budget, dim, population_size=50, explosion_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.explosion_rate = explosion_rate

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        best_solution = population[np.argmin([func(individual) for individual in population])]
        for _ in range(self.budget):
            fireworks = []
            for i in range(self.population_size):
                explosion_size = int(self.explosion_rate * self.population_size)
                for _ in range(explosion_size):
                    fireworks.append(population[i] + np.random.normal(0, 1, self.dim))
            
            fireworks = np.array(fireworks)
            fireworks_costs = [func(firework) for firework in fireworks]
            min_firework = fireworks[np.argmin(fireworks_costs)]

            if func(min_firework) < func(best_solution):
                best_solution = min_firework

        return best_solution