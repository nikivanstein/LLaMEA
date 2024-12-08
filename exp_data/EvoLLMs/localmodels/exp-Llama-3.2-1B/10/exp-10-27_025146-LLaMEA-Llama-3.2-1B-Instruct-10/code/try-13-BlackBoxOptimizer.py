# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a random point in the search space
            new_individual = copy.deepcopy(self.search_space[np.random.randint(0, self.dim)])

            # Evaluate the function at the current point
            value = func(new_individual)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = new_individual

        # Return the optimized value
        return best_value

    def mutate(self, individual):
        """
        Mutate the individual by changing a random value.

        Args:
            individual (list): The individual to mutate.

        Returns:
            list: The mutated individual.
        """
        # Randomly select a value to mutate
        mutation_index = np.random.randint(0, self.dim)

        # Change the value at the selected index
        individual[mutation_index] = np.random.uniform(-5.0, 5.0)

        return individual

    def evolve(self, population_size, mutation_rate):
        """
        Evolve the population using mutation and selection.

        Args:
            population_size (int): The size of the population.
            mutation_rate (float): The probability of mutation.

        Returns:
            list: The evolved population.
        """
        # Initialize the population with random individuals
        population = [copy.deepcopy(self.search_space[np.random.randint(0, self.dim)]) for _ in range(population_size)]

        # Perform selection and mutation
        while len(population) < self.budget:
            # Select the fittest individual
            fittest_individual = max(population, key=self.__call__)

            # Mutate the fittest individual
            mutated_individual = self.mutate(fittest_individual)

            # Add the mutated individual to the population
            population.append(mutated_individual)

        return population

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# Novel Metaheuristic Algorithm for Black Box Optimization
# ```
# ```python
# import random
# import numpy as np
# import copy

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the NovelMetaheuristicOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        """
        Optimize the black box function using the NovelMetaheuristicOptimizer.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a random point in the search space
            new_individual = copy.deepcopy(self.search_space[np.random.randint(0, self.dim)])

            # Evaluate the function at the current point
            value = func(new_individual)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = new_individual

        # Return the optimized value
        return best_value

    def mutate(self, individual):
        """
        Mutate the individual by changing a random value.

        Args:
            individual (list): The individual to mutate.

        Returns:
            list: The mutated individual.
        """
        # Randomly select a value to mutate
        mutation_index = np.random.randint(0, self.dim)

        # Change the value at the selected index
        individual[mutation_index] = np.random.uniform(-5.0, 5.0)

        return individual

    def evolve(self, population_size, mutation_rate):
        """
        Evolve the population using mutation and selection.

        Args:
            population_size (int): The size of the population.
            mutation_rate (float): The probability of mutation.

        Returns:
            list: The evolved population.
        """
        # Initialize the population with random individuals
        population = [copy.deepcopy(self.search_space[np.random.randint(0, self.dim)]) for _ in range(population_size)]

        # Perform selection and mutation
        while len(population) < self.budget:
            # Select the fittest individual
            fittest_individual = max(population, key=self.__call__)

            # Mutate the fittest individual
            mutated_individual = self.mutate(fittest_individual)

            # Add the mutated individual to the population
            population.append(mutated_individual)

        return population

# Example usage
budget = 100
dim = 5
optimizer = NovelMetaheuristicOptimizer(budget, dim)
optimized_function = optimizer(__call__)

# Print the optimized function
print("Optimized function:", optimized_function)

# Print the fitness score
print("Fitness score:", optimized_function)

# Print the evolution history
history = optimizer.evolve(100, 0.1)
print("Evolution history:")
for individual in history:
    print(individual)