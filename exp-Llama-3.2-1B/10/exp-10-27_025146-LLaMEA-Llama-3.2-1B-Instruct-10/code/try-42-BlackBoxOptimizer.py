# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from copy import deepcopy
from collections import deque

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
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = point

        # Return the optimized value
        return best_value

    def mutate(self, individual):
        """
        Mutate the given individual by changing a random bit.

        Args:
            individual (List[float]): The individual to mutate.

        Returns:
            List[float]: The mutated individual.
        """
        # Create a copy of the individual
        mutated_individual = deepcopy(individual)

        # Randomly select a bit to mutate
        index = np.random.randint(0, self.dim)

        # Flip the bit
        mutated_individual[index] = 1 - mutated_individual[index]

        return mutated_individual

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to generate a child.

        Args:
            parent1 (List[float]): The first parent.
            parent2 (List[float]): The second parent.

        Returns:
            List[float]: The child.
        """
        # Create a copy of the parents
        child = deepcopy(parent1)

        # Randomly select a sub-range to crossover
        start = np.random.randint(0, self.dim)
        end = np.random.randint(start, self.dim)

        # Crossover the sub-ranges
        child[start:end] = np.concatenate((parent1[start:end], parent2[start:end]))

        return child

# One-line description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
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
            # Initialize the current best value and its corresponding index
            current_best_value = float('-inf')
            current_best_index = -1

            # Initialize the mutation and crossover rates
            mutation_rate = 0.1
            crossover_rate = 0.1

            # Initialize the queue with the initial individual
            queue = deque([self._initialize_individual(func)])

            while queue:
                # Dequeue the individual
                individual = queue.popleft()

                # Evaluate the function at the current individual
                value = func(individual)

                # If the current value is better than the best value found so far,
                # update the best value and its corresponding index
                if value > current_best_value:
                    current_best_value = value
                    current_best_index = individual

                # If the current value is better than the best value found so far,
                # and a mutation or crossover has been performed,
                # update the best value and its corresponding index
                if value > best_value and random.random() < mutation_rate:
                    # Mutate the individual
                    individual = self._mutate(individual)

                    # Evaluate the function at the mutated individual
                    mutated_value = func(individual)

                    # If the mutated value is better than the best value found so far,
                    # update the best value and its corresponding index
                    if mutated_value > current_best_value:
                        current_best_value = mutated_value
                        current_best_index = individual

                # If a crossover has been performed,
                # generate two parents and crossover them
                if random.random() < crossover_rate:
                    parent1 = self._crossover(parent1, parent2)
                    parent2 = self._crossover(parent1, parent2)

                    # Evaluate the function at the parents
                    value1 = func(parent1)
                    value2 = func(parent2)

                    # If the parents have different values, select one
                    if value1!= value2:
                        # Select one of the parents
                        individual = parent1 if value1 > value2 else parent2

                        # Evaluate the function at the selected individual
                        value = func(individual)

                        # If the selected value is better than the current best value,
                        # update the best value and its corresponding index
                        if value > current_best_value:
                            current_best_value = value
                            current_best_index = individual

                # Add the individual to the queue
                queue.append(individual)

        # Return the optimized value
        return current_best_value

    def _initialize_individual(self, func):
        """
        Initialize the initial individual with a random point in the search space.

        Args:
            func (callable): The black box function.

        Returns:
            List[float]: The initial individual.
        """
        return np.random.uniform(-5.0, 5.0, self.dim)

    def _mutate(self, individual):
        """
        Mutate the given individual by changing a random bit.

        Args:
            individual (List[float]): The individual to mutate.

        Returns:
            List[float]: The mutated individual.
        """
        # Create a copy of the individual
        mutated_individual = deepcopy(individual)

        # Randomly select a bit to mutate
        index = np.random.randint(0, self.dim)

        # Flip the bit
        mutated_individual[index] = 1 - mutated_individual[index]

        return mutated_individual

    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to generate a child.

        Args:
            parent1 (List[float]): The first parent.
            parent2 (List[float]): The second parent.

        Returns:
            List[float]: The child.
        """
        # Create a copy of the parents
        child = deepcopy(parent1)

        # Randomly select a sub-range to crossover
        start = np.random.randint(0, self.dim)
        end = np.random.randint(start, self.dim)

        # Crossover the sub-ranges
        child[start:end] = np.concatenate((parent1[start:end], parent2[start:end]))

        return child

# One-line description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# import random
# import numpy as np

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
            # Initialize the current best value and its corresponding index
            current_best_value = float('-inf')
            current_best_index = -1

            # Initialize the mutation and crossover rates
            mutation_rate = 0.1
            crossover_rate = 0.1

            # Initialize the queue with the initial individual
            queue = deque([self._initialize_individual(func)])

            while queue:
                # Dequeue the individual
                individual = queue.popleft()

                # Evaluate the function at the current individual
                value = func(individual)

                # If the current value is better than the best value found so far,
                # update the best value and its corresponding index
                if value > current_best_value:
                    current_best_value = value
                    current_best_index = individual

                # If the current value is better than the best value found so far,
                # and a mutation or crossover has been performed,
                # update the best value and its corresponding index
                if value > best_value and random.random() < mutation_rate:
                    # Mutate the individual
                    individual = self._mutate(individual)

                    # Evaluate the function at the mutated individual
                    mutated_value = func(individual)

                    # If the mutated value is better than the best value found so far,
                    # update the best value and its corresponding index
                    if mutated_value > current_best_value:
                        current_best_value = mutated_value
                        current_best_index = individual

                # If a crossover has been performed,
                # generate two parents and crossover them
                if random.random() < crossover_rate:
                    parent1 = self._crossover(parent1, parent2)
                    parent2 = self._crossover(parent1, parent2)

                    # Evaluate the function at the parents
                    value1 = func(parent1)
                    value2 = func(parent2)

                    # If the parents have different values, select one
                    if value1!= value2:
                        # Select one of the parents
                        individual = parent1 if value1 > value2 else parent2

                        # Evaluate the function at the selected individual
                        value = func(individual)

                        # If the selected value is better than the current best value,
                        # update the best value and its corresponding index
                        if value > current_best_value:
                            current_best_value = value
                            current_best_index = individual

                # Add the individual to the queue
                queue.append(individual)

        # Return the optimized value
        return current_best_value

# Usage
budget = 100
dim = 5
func = lambda x: x**2
optimizer = NovelMetaheuristicOptimizer(budget, dim)
optimized_value = optimizer(func)

print("Optimized value:", optimized_value)