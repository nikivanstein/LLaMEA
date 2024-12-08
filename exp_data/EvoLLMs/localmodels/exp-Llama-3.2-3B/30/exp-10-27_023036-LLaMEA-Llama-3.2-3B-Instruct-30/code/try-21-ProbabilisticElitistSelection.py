import numpy as np
import random

class ProbabilisticElitistSelection:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = []
        self.best_individual = None

    def __call__(self, func):
        # Initialize population with random individuals
        for _ in range(self.budget):
            self.population.append([random.uniform(self.lower_bound, self.upper_bound) for _ in range(self.dim)])

        # Evaluate population and update best individual
        for individual in self.population:
            value = func(individual)
            if self.best_individual is None or value < func(self.best_individual):
                self.best_individual = individual

        # Select and refine the best individual
        if self.best_individual is not None:
            # Calculate probability of changing each line of the best individual
            probabilities = [0.3] * self.dim
            new_individual = []
            for i in range(self.dim):
                if random.random() < probabilities[i]:
                    new_individual.append(random.uniform(self.lower_bound, self.upper_bound))
                else:
                    new_individual.append(self.best_individual[i])
            self.population[0] = new_individual

        # Repeat until budget is reached
        while len(self.population) < self.budget:
            # Select and refine the best individual
            if self.best_individual is not None:
                # Calculate probability of changing each line of the best individual
                probabilities = [0.3] * self.dim
                new_individual = []
                for i in range(self.dim):
                    if random.random() < probabilities[i]:
                        new_individual.append(random.uniform(self.lower_bound, self.upper_bound))
                    else:
                        new_individual.append(self.best_individual[i])
                self.population.append(new_individual)
            else:
                # Create a new individual
                new_individual = [random.uniform(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
                self.population.append(new_individual)

            # Evaluate new individual and update best individual
            value = func(new_individual)
            if value < func(self.best_individual):
                self.best_individual = new_individual

# Example usage
if __name__ == "__main__":
    func = lambda x: sum([i**2 for i in x])  # Example function
    optimizer = ProbabilisticElitistSelection(100, 10)
    optimizer(func)
    print("Best individual:", optimizer.best_individual)
    print("Best function value:", func(optimizer.best_individual))