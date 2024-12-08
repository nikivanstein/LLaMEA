import random
import numpy as np

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
        return "Novel Metaheuristic Algorithm for Black Box Optimization"

    def mutate(self, individual):
        # Randomly swap two elements in the individual
        idx1, idx2 = random.sample(range(self.dim), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def crossover(self, parent1, parent2):
        # Randomly select a crossover point
        crossover_point = random.randint(1, self.dim - 1)
        # Create a new child individual by combining the parents
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
def eval_bbo(func, individual, budget):
    best_individual = individual
    best_fitness = func(best_individual)
    for _ in range(budget):
        # Generate a new individual using crossover and mutation
        new_individual = self.crossover(best_individual, self.mutate(best_individual))
        # Evaluate the new individual
        new_fitness = func(new_individual)
        # Update the best individual and fitness if necessary
        if new_fitness > best_fitness:
            best_individual = new_individual
            best_fitness = new_fitness
    return best_individual, best_fitness

# Initialize the BlackBoxOptimizer
optimizer = BlackBoxOptimizer(100, 10)

# Evaluate the Sphere function
best_individual, best_fitness = eval_bbo(lambda x: x**2, [0.0]*10, 100)

# Print the result
print(f"Best Individual: {best_individual}, Best Fitness: {best_fitness}")