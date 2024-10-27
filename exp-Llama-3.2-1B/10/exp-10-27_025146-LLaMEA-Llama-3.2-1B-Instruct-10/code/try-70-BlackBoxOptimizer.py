# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import minimize
from typing import List

class BlackBoxOptimizer:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func: callable) -> float:
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

    def _mutation(self, individual: List[float], mutation_prob: float) -> List[float]:
        """
        Apply a mutation to an individual.

        Args:
            individual (List[float]): The individual to mutate.
            mutation_prob (float): The probability of mutation.

        Returns:
            List[float]: The mutated individual.
        """
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if random.random() < mutation_prob:
                mutated_individual[i] += np.random.uniform(-1, 1)
                mutated_individual[i] = max(-5.0, min(5.0, mutated_individual[i]))
        return mutated_individual

    def _crossover(self, parents: List[List[float]], mutation_prob: float) -> List[List[float]]:
        """
        Perform crossover between two parents.

        Args:
            parents (List[List[float]]): The parents to crossover.
            mutation_prob (float): The probability of mutation.

        Returns:
            List[List[float]]: The offspring.
        """
        offspring = []
        for _ in range(self.budget // 2):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = parent1[:self.dim]
            child += self._mutation(parent2, mutation_prob)
            offspring.append(child)
        return offspring

    def _select(self, parents: List[List[float]], mutation_prob: float) -> List[List[float]]:
        """
        Select the parents for crossover.

        Args:
            parents (List[List[float]]): The parents to select.
            mutation_prob (float): The probability of mutation.

        Returns:
            List[List[float]]: The selected parents.
        """
        selected_parents = []
        for _ in range(self.budget // 2):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            selected_parent = self._crossover(parent1, mutation_prob)
            selected_parent = self._select(selected_parent, mutation_prob)
            selected_parents.append(selected_parent)
        return selected_parents

    def optimize(self, func: callable) -> float:
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

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import minimize
from typing import List

class BlackBoxOptimizer:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func: callable) -> float:
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

    def _mutation(self, individual: List[float], mutation_prob: float) -> List[float]:
        """
        Apply a mutation to an individual.

        Args:
            individual (List[float]): The individual to mutate.
            mutation_prob (float): The probability of mutation.

        Returns:
            List[float]: The mutated individual.
        """
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if random.random() < mutation_prob:
                mutated_individual[i] += np.random.uniform(-1, 1)
                mutated_individual[i] = max(-5.0, min(5.0, mutated_individual[i]))
        return mutated_individual

    def _crossover(self, parents: List[List[float]], mutation_prob: float) -> List[List[float]]:
        """
        Perform crossover between two parents.

        Args:
            parents (List[List[float]]): The parents to crossover.
            mutation_prob (float): The probability of mutation.

        Returns:
            List[List[float]]: The offspring.
        """
        offspring = []
        for _ in range(self.budget // 2):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = parent1[:self.dim]
            child += self._mutation(parent2, mutation_prob)
            offspring.append(child)
        return offspring

    def _select(self, parents: List[List[float]], mutation_prob: float) -> List[List[float]]:
        """
        Select the parents for crossover.

        Args:
            parents (List[List[float]]): The parents to select.
            mutation_prob (float): The probability of mutation.

        Returns:
            List[List[float]]: The selected parents.
        """
        selected_parents = []
        for _ in range(self.budget // 2):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            selected_parent = self._crossover(parent1, mutation_prob)
            selected_parent = self._select(selected_parent, mutation_prob)
            selected_parents.append(selected_parent)
        return selected_parents

    def optimize(self, func: callable) -> float:
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

# Example usage:
def func(x: List[float]) -> float:
    return sum(x) / len(x)

optimizer = BlackBoxOptimizer(100, 5)
best_value = optimizer.optimize(func)

print(f"Best value: {best_value}")
print(f"Best index: {best_index}")