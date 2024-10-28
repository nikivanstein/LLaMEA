import numpy as np
from scipy.optimize import minimize

class EvolutionaryTreeOfSubspace:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population_size = 10
        self.tree_size = 10
        self.refinement_probability = 0.4

    def __call__(self, func):
        if self.budget <= 0:
            return None

        # Initialize the population
        population = np.random.uniform(self.search_space[0], self.search_space[-1], (self.population_size, self.dim))

        for _ in range(self.budget):
            # Evaluate the population
            evaluations = [func(x) for x in population]

            # Select the fittest individuals
            fittest_indices = np.argsort(evaluations)[:self.tree_size]
            fittest_population = population[fittest_indices]

            # Create a new generation
            new_generation = []
            for _ in range(self.population_size):
                # Select a parent from the fittest population
                parent = np.random.choice(fittest_population, 1)[0]

                # Create a child by perturbing the parent with refinement probability
                child = parent + np.random.uniform(-self.search_space[0], self.search_space[-1], self.dim)
                if np.random.rand() < self.refinement_probability:
                    # Refine the child by perturbing its dimensions
                    child = child + np.random.uniform(-self.search_space[0], self.search_space[-1], self.dim)

                # Add the child to the new generation
                new_generation.append(child)

            # Update the population
            population = np.array(new_generation)

        # Return the best individual
        best_individual = population[np.argmin([func(x) for x in population])]
        return best_individual

# Example usage
def func(x):
    return x[0]**2 + x[1]**2 + x[2]**2

budget = 100
dim = 3
optimization_algorithm = EvolutionaryTreeOfSubspace(budget, dim)
best_individual = optimization_algorithm(func)
print(best_individual)