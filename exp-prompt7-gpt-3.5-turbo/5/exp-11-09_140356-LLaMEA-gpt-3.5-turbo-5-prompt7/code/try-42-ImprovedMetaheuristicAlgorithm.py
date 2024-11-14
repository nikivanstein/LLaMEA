import numpy as np

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.learning_rate = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            # Select parents
            idx = np.argsort([func(x) for x in self.population])
            parent1, parent2 = self.population[idx[0]], self.population[idx[1]]

            # Adaptive learning rate mutation strategy
            beta = np.random.uniform(0.5, 1.0, self.dim) * self.learning_rate
            offspring = parent1 + beta * (parent2 - self.population)

            # Replace worst solution
            idx_worst = np.argmax([func(x) for x in self.population])
            self.population[idx_worst] = offspring

            # Update learning rate based on improvement
            best_solution = self.population[idx[0]]
            improvement = func(best_solution) - func(parent1)
            self.learning_rate *= 1.1 if improvement > 0 else 0.9

        return self.population[idx[0]]