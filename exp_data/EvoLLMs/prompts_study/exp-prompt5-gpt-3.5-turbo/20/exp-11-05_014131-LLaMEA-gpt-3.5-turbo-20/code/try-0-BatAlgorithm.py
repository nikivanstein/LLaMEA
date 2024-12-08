import numpy as np

class BatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_individual = population[np.argmin(fitness)]
        
        t = 0
        while t < self.budget:
            for i in range(self.population_size):
                frequency = np.random.uniform(-1, 1, self.dim)
                velocity = population[i] + (population[i] - best_individual) * self.alpha + frequency * self.gamma
                new_solution = population[i] + velocity
                if np.random.rand() < self.pulse_rate:
                    new_solution = best_individual + np.random.uniform(-1, 1, self.dim) * self.loudness
                
                new_fitness = func(new_solution)
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
                    if new_fitness < func(best_individual):
                        best_individual = new_solution

                t += 1
                if t >= self.budget:
                    break

        return best_individual