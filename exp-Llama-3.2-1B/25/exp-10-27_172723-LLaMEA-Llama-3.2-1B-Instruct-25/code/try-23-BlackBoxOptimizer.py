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
        # Select two random individuals from the population
        parent1, parent2 = random.sample(self.population, 2)

        # Select a random crossover point
        crossover_point = random.randint(1, self.dim - 1)

        # Perform crossover
        child = [x for i, x in enumerate(parent1) if i < crossover_point] + [x for x in parent2 if x > crossover_point] + [x for x in parent1[i + crossover_point:] if x < crossover_point]

        # Select a random mutation point
        mutation_point = random.randint(0, self.dim - 1)

        # Perform mutation
        child[mutation_point] = random.uniform(-5.0, 5.0)

        # Replace the original individual with the mutated individual
        self.population[self.population.index(individual)] = child

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"