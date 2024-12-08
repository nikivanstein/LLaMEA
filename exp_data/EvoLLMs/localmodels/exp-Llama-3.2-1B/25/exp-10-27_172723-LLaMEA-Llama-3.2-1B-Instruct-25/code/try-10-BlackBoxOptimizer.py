import random
import numpy as np
from scipy.optimize import differential_evolution

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

    def evolve(self, func):
        # Define the mutation rate
        mutation_rate = 0.01

        # Define the crossover rate
        crossover_rate = 0.5

        # Initialize the population
        population = [func(np.random.rand(self.dim)) for _ in range(100)]

        # Evolve the population
        for _ in range(100):
            # Select parents using tournament selection
            parents = random.sample(population, len(population) // 2)

            # Evaluate the fitness of each parent
            fitness = [self(func(x)) for x in parents]

            # Select the fittest parents
            fittest_parents = np.array([parents[i] for i, f in enumerate(fitness) if f == max(fitness)])

            # Crossover the fittest parents
            children = []
            for _ in range(len(fittest_parents)):
                parent1, parent2 = fittest_parents[_*2:(_+1)*2]
                child = (parent1 + parent2) / 2
                if random.random() < crossover_rate:
                    # Mutate the child
                    child = self.evaluate_fitness(child)
                children.append(child)

            # Mutate the children
            children = [self.evaluate_fitness(x) for x in children]

            # Replace the old population with the new one
            population = children

        # Return the fittest individual
        return population[np.argmax([self.evaluate_fitness(x) for x in population])]

# One-line description: "Evolutionary Algorithm: A novel metaheuristic algorithm that efficiently solves black box optimization problems by combining evolutionary strategies with function evaluation"