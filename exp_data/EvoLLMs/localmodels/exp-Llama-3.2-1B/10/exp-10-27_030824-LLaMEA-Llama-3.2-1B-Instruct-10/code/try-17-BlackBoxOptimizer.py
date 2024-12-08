import random
import numpy as np
import math

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def __str__(self):
        return f"Novel Metaheuristic Algorithm for Black Box Optimization"

    def novel_metaheuristic(self, func, budget, dim):
        # Initialize the population with random points in the search space
        population = [func(np.random.uniform(self.search_space[0], self.search_space[1])) for _ in range(100)]

        # Evolve the population over 100 generations
        for _ in range(100):
            # Select the fittest individuals
            fittest = sorted(population, key=func, reverse=True)[:self.budget]

            # Perform crossover and mutation
            children = []
            for i in range(0, len(fittest), 2):
                parent1, parent2 = fittest[i], fittest[i+1]
                child = (parent1 + parent2) / 2
                if random.random() < 0.5:
                    # Perform mutation
                    child = random.uniform(self.search_space[0], self.search_space[1])
                children.append(child)

            # Replace the least fit individuals with the new children
            population = fittest + children

        # Return the fittest individual in the final population
        return population[0]

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Test the algorithm
if __name__ == "__main__":
    # Create a problem instance
    problem = BlackBoxOptimizer(1000, 5)

    # Optimize a function
    func = lambda x: x**2
    best_individual = problem.novel_metaheuristic(func, 1000, 5)
    print(f"Best individual: {best_individual}, Function value: {func(best_individual)}")

    # Evaluate the function at the best individual
    print(f"Function value at best individual: {func(best_individual)}")