import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.best_individual = None

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

    def mutate(self, individual):
        # Randomly select a mutation point
        mutation_point = np.random.randint(0, self.dim)
        # Swap the values at the mutation point
        individual[mutation_point], individual[mutation_point + 1] = individual[mutation_point + 1], individual[mutation_point]
        # Return the mutated individual
        return individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = np.random.randint(0, self.dim)
        # Create a new individual by combining the parents
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        # Return the child individual
        return child

    def __repr__(self):
        return "Novel Metaheuristic Algorithm for Black Box Optimization"

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

def evaluateBBOB(func, budget):
    algorithm = BlackBoxOptimizer(budget, 5)
    for _ in range(1000):
        # Generate a new individual
        individual = np.random.uniform(-5.0, 5.0, self.dim)
        # Evaluate the function at the individual
        evaluation = func(individual)
        # If the evaluation exceeds the budget, return a default point and evaluation
        if evaluation > budget:
            return np.random.uniform(-5.0, 5.0), func(np.random.uniform(-5.0, 5.0))
        # If the individual is better than the current best, update the best individual
        if evaluation < algorithm.best_individual[0]:
            algorithm.best_individual = (individual, evaluation)
    # Return the best individual and evaluation
    return algorithm.best_individual[0], algorithm.best_individual[1]

# Evaluate the BBOB test suite
for func in ["sphere", "rosett", "sphere-2", "rosett-2", "sphere-3", "rosett-3", "sphere-4", "rosett-4"]:
    budget = 1000
    print(f"Evaluating {func} with budget {budget}")
    evaluation = evaluateBBOB(func, budget)
    print(f"Best individual: {evaluation}")