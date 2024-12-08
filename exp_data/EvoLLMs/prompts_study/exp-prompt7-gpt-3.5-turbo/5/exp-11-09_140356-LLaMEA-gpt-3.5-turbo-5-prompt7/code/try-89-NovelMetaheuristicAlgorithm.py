import numpy as np

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.step_size = 0.2

    def __call__(self, func):
        for _ in range(self.budget):
            idx = np.argsort([func(x) for x in self.population])
            parent1, parent2 = self.population[idx[0]], self.population[idx[1]]
            
            beta = np.random.uniform(0.5, 1.0, self.dim)
            offspring = parent1 + self.step_size * beta * (parent2 - self.population)
            
            idx_worst = np.argmax([func(x) for x in self.population])
            self.population[idx_worst] = offspring
            
            # Dynamically adjust step size based on solution quality
            if func(offspring) < func(self.population[idx_worst]):
                self.step_size *= 1.1
            else:
                self.step_size *= 0.9

        return self.population[idx[0]]