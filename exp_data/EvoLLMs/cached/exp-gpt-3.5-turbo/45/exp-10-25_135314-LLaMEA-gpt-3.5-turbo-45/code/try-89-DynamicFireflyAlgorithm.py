import numpy as np

class DynamicFireflyAlgorithm:
    def __init__(self, budget, dim, population_size=20, alpha_min=0.1, alpha_max=0.9, gamma_min=0.1, gamma_max=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def attractiveness(self, r):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * np.exp(-r)

    def move_firefly(self, fireflies, current_firefly, iteration):
        for i in range(self.dim):
            r = np.linalg.norm(fireflies - current_firefly, axis=1)
            beta = self.attractiveness(r)
            gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * (iteration / self.budget)
            move = gamma * (np.random.rand() - 0.5) + beta * (fireflies[:, i] - current_firefly[i])
            current_firefly[i] += move
            current_firefly[i] = np.clip(current_firefly[i], -5.0, 5.0)
        return current_firefly

    def __call__(self, func):
        fireflies = self.initialize_population()
        firefly_fitness = np.array([func(individual) for individual in fireflies])

        for iteration in range(self.budget):
            for idx, firefly in enumerate(fireflies):
                new_firefly = self.move_firefly(fireflies, firefly, iteration)
                new_fitness = func(new_firefly)
                
                if new_fitness < firefly_fitness[idx]:
                    fireflies[idx] = new_firefly
                    firefly_fitness[idx] = new_fitness
        
        return fireflies[np.argmin(firefly_fitness)]