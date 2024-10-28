import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any

class BlackBoxOptimizer:
    def __init__(self, budget: int, dim: int, func: Dict[str, float]) -> None:
        """
        Initialize the BlackBoxOptimizer with a given budget, dimension, and a black box function.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.
        """
        self.budget = budget
        self.dim = dim
        self.func = func

    def __call__(self, func: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize the black box function using a novel heuristic algorithm.

        Args:
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.

        Returns:
        Dict[str, float]: The optimized function values.
        """
        # Initialize the search space with random values
        x = np.random.uniform(-5.0, 5.0, self.dim)

        # Define the objective function to minimize (negative of the original function)
        def objective(x: np.ndarray) -> float:
            return -np.sum(self.func.values(x))

        # Define the bounds for the search space
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]

        # Define the mutation rate
        mutation_rate = 0.01

        # Define the crossover rate
        crossover_rate = 0.5

        # Initialize the population
        population = self.generate_initial_population(self.budget, self.dim)

        # Run the evolutionary algorithm
        for _ in range(100):  # Run for 100 generations
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=lambda x: self.evaluate_fitness(x), reverse=True)[:self.budget]

            # Create a new generation
            new_population = self.generate_new_population(fittest_individuals, mutation_rate, bounds, crossover_rate)

            # Replace the old population with the new one
            population = new_population

        # Return the optimized function values
        return {k: -v for k, v in self.evaluate_fitness(population)}

    def generate_initial_population(self, budget: int, dim: int) -> List[Dict[str, float]]:
        """
        Generate an initial population of random individuals.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.

        Returns:
        List[Dict[str, float]]: The initial population of individuals.
        """
        return [self.generate_individual(dim) for _ in range(budget)]

    def generate_new_population(self, fittest_individuals: List[Dict[str, float]], mutation_rate: float, bounds: List[float], crossover_rate: float) -> List[Dict[str, float]]:
        """
        Generate a new population of individuals by crossover and mutation.

        Args:
        fittest_individuals (List[Dict[str, float]]): The fittest individuals in the population.
        mutation_rate (float): The mutation rate.
        bounds (List[float]): The bounds for the search space.
        crossover_rate (float): The crossover rate.

        Returns:
        List[Dict[str, float]]: The new population of individuals.
        """
        new_population = []

        for individual in fittest_individuals:
            # Select two parents using tournament selection
            parent1 = self.select_parent(individual, fittest_individuals)
            parent2 = self.select_parent(individual, fittest_individuals)

            # Perform crossover
            child = self.crossover(parent1, parent2, mutation_rate, bounds)

            # Perform mutation
            child = self.mutate(child, mutation_rate, bounds)

            new_population.append(child)

        return new_population

    def select_parent(self, individual: Dict[str, float], fittest_individuals: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Select a parent using tournament selection.

        Args:
        individual (Dict[str, float]): The individual to select.
        fittest_individuals (List[Dict[str, float]]): The fittest individuals in the population.

        Returns:
        Dict[str, float]: The selected parent.
        """
        # Get the values of the individual
        values = [individual[k] for k in individual]

        # Select the top k values
        top_k = sorted(enumerate(values), key=lambda x: x[1], reverse=True)[:len(fittest_individuals)]

        # Select the top k parents
        parent1 = fittest_individuals[top_k[0]]
        parent2 = fittest_individuals[top_k[1]]

        return parent1, parent2

    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float], mutation_rate: float, bounds: List[float]) -> Dict[str, float]:
        """
        Perform crossover between two parents.

        Args:
        parent1 (Dict[str, float]): The first parent.
        parent2 (Dict[str, float]): The second parent.
        mutation_rate (float): The mutation rate.
        bounds (List[float]): The bounds for the search space.

        Returns:
        Dict[str, float]: The child individual.
        """
        # Get the values of the parents
        values1 = [parent1[k] for k in parent1]
        values2 = [parent2[k] for k in parent2]

        # Perform crossover
        child = {}

        for k in values1:
            if np.random.rand() < mutation_rate:
                child[k] = np.random.uniform(bounds[0], bounds[1])
            else:
                child[k] = values1[k]

        for k in values2:
            if np.random.rand() < mutation_rate:
                child[k] = np.random.uniform(bounds[0], bounds[1])
            else:
                child[k] = values2[k]

        return child

    def mutate(self, individual: Dict[str, float], mutation_rate: float, bounds: List[float]) -> Dict[str, float]:
        """
        Perform mutation on an individual.

        Args:
        individual (Dict[str, float]): The individual to mutate.
        mutation_rate (float): The mutation rate.
        bounds (List[float]): The bounds for the search space.

        Returns:
        Dict[str, float]: The mutated individual.
        """
        # Get the values of the individual
        values = [individual[k] for k in individual]

        # Perform mutation
        for i in range(len(values)):
            if np.random.rand() < mutation_rate:
                values[i] = np.random.uniform(bounds[0], bounds[1])

        return {k: v for k, v in individual.items() if k in values}