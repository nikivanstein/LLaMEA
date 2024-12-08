import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a given budget and dimensionality.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size
        population_size = 100

        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

        # Define a mutation function to introduce random changes in the population
        def mutate(individual):
            mutated_individual = individual.copy()
            np.random.shuffle(individual)
            mutated_individual[:self.dim] = np.random.uniform(-5.0, 5.0, self.dim)
            return mutated_individual

        # Define a selection function to select the fittest individuals
        def select(fittest_individuals, population_size):
            return np.argsort(population_size * func_values[fittest_individuals])[:population_size]

        # Define a crossover function to combine two individuals
        def crossover(parent1, parent2):
            child = parent1.copy()
            np.random.shuffle(child)
            child[:self.dim] = np.random.uniform(-5.0, 5.0, self.dim)
            return child

        # Define a mutation rate
        mutation_rate = 0.05

        # Perform selection, crossover, and mutation to optimize the function
        for _ in range(self.budget):
            # Select the fittest individuals
            fittest_individuals = select(fittest_individuals, population_size)

            # Perform crossover to combine two individuals
            child = crossover(fittest_individuals[0], fittest_individuals[1])

            # Perform mutation on the child
            mutated_individual = mutate(child)

            # Replace the old population with the new population
            population = np.concatenate([population, mutated_individual])

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# Example usage:
if __name__ == "__main__":
    # Create a BlackBoxOptimizer with a budget of 1000 evaluations and a dimensionality of 10
    optimizer = BlackBoxOptimizer(1000, 10)

    # Optimize a black box function using the optimizer
    func = lambda x: np.sin(x)
    optimized_params, optimized_func_value = optimizer(func)
    print("Optimized parameters:", optimized_params)
    print("Optimized function value:", optimized_func_value)