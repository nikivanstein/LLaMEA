import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def mutate(self, individual):
        # Select a random individual from the current population
        parent1, parent2 = random.sample(self.population, 2)

        # Create a crossover offspring
        child = np.concatenate((parent1[:self.dim // 2], parent2[self.dim // 2:]))

        # Create a mutation offspring
        mutated_child = np.random.uniform(child, size=self.dim)

        # Replace the parent with the mutated offspring
        self.population[(self.population.index(parent1) + 1):] = [mutated_child]

        # Return the mutated individual
        return mutated_child

    def evaluate_fitness(self, individual, problem):
        # Evaluate the function at the individual
        value = problem.evaluate(individual)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current individual as the optimal solution
            return individual
        else:
            # If the function has been evaluated within the budget, return the individual
            return individual

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"