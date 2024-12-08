import numpy as np

class SLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_scores = []

    def __call__(self, func):
        # Initialize the population with random solutions
        for _ in range(100):  # arbitrary initial population size
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            self.population.append(solution)

        # Evaluate the function for each solution and store the fitness scores
        for i, solution in enumerate(self.population):
            func(solution)
            self.fitness_scores.append(self.fitness_score(func))

        # Select the best solutions based on the fitness scores
        self.population = self.select_best_solutions(self.fitness_scores)

        # Perform stochastic local search to refine the best solutions
        for _ in range(self.budget):
            best_solution = self.population[np.argmax(self.fitness_scores)]
            for i in range(self.dim):
                # Randomly perturb the solution
                perturbed_solution = best_solution + np.random.uniform(-0.1, 0.1, self.dim)
                # Check if the perturbed solution is within the search space
                if -5.0 <= perturbed_solution <= 5.0:
                    # Evaluate the function for the perturbed solution
                    func(perturbed_solution)
                    # Update the best solution if the function value is higher
                    if self.fitness_score(func) > self.fitness_scores[i]:
                        self.population[i] = perturbed_solution
                        self.fitness_scores[i] = self.fitness_score(func)

        return self.population[0]

    def fitness_score(self, func):
        # Evaluate the function at a given solution
        return func(self.population[0])

    def select_best_solutions(self, fitness_scores):
        # Select the solutions with the highest fitness scores
        return self.population[np.argsort(-fitness_scores)]

# Example usage:
from blackboxopt import bbopt

# Define the black box function
def func(x):
    return x**2 + 2*x + 1

# Create an instance of the SLS algorithm
sls = SLS(budget=100, dim=2)

# Optimize the function using the SLS algorithm
best_solution = sls(func)

# Print the best solution
print("Best solution:", best_solution)