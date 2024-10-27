import numpy as np
from scipy.optimize import minimize
from collections import deque

class BlackBoxOptimizer:
    """
    An optimization algorithm that handles a wide range of tasks by leveraging the power of black box optimization.

    Attributes:
    ----------
    budget : int
        The maximum number of function evaluations allowed.
    dim : int
        The dimensionality of the search space.

    Methods:
    -------
    __init__(self, budget, dim)
        Initializes the optimization algorithm with the given budget and dimensionality.
    def __call__(self, func)
        Optimizes the black box function `func` using `self.budget` function evaluations.
    def evaluate_fitness(self, individual):
        """
        Evaluates the fitness of the given individual using the given function.

        Parameters:
        ----------
        individual : Individual
            The individual to evaluate.

        Returns:
        -------
        float
            The fitness value of the individual.
        """
        # Define the bounds for the search space
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]

        # Initialize the parameters with random values within the bounds
        params = np.random.uniform(-5.0, 5.0, self.dim)

        # Define the function to minimize (in this case, the negative of the function value)
        def neg_func(params):
            return -func(params)

        # Use the minimize function from SciPy to optimize the function
        result = minimize(neg_func, params, method="SLSQP", bounds=bounds, options={"maxiter": self.budget})

        # Return the optimized parameters and the optimized function value
        return result.x, -result.fun

    def __next_solution(self, individual, population_size):
        """
        Selects the next individual from the population using the probability of 0.2.

        Parameters:
        ----------
        individual : Individual
            The individual to select.
        population_size : int
            The size of the population.

        Returns:
        -------
        Individual
            The selected individual.
        """
        # Select the next individual from the population using the probability of 0.2
        selected_individual = np.random.choice([individual, individual[::-1]], p=[0.8, 0.2])

        # Refine the selected individual's strategy by changing its lines of code
        for i in range(len(selected_individual)):
            if selected_individual[i] == 0:
                selected_individual[i] = 1 - selected_individual[i]

        return selected_individual

    def __next_population(self, population_size):
        """
        Selects the next population of individuals using the probability of 0.2.

        Parameters:
        ----------
        population_size : int
            The size of the population.

        Returns:
        -------
        list
            The selected population of individuals.
        """
        # Select the next population of individuals using the probability of 0.2
        selected_population = [self.__next_solution(individual, population_size) for individual in np.random.choice([individual, individual[::-1]], size=population_size, replace=True)]

        return selected_population


# Example usage:
if __name__ == "__main__":
    # Initialize the optimization algorithm with a budget of 1000 function evaluations
    optimizer = BlackBoxOptimizer(budget=1000, dim=5)

    # Define the black box function to optimize
    def func(params):
        return np.sum(params ** 2)

    # Initialize the population of individuals
    population = np.random.uniform(-5.0, 5.0, (100, 5))

    # Initialize the best solution and its fitness
    best_solution = None
    best_fitness = float('-inf')

    # Run the optimization algorithm
    for _ in range(100):
        # Evaluate the fitness of the current population
        fitness = [optimizer.evaluate_fitness(individual) for individual in population]

        # Select the next population of individuals
        next_population = optimizer.__next_population(population_size=100)

        # Update the best solution and its fitness
        new_solution = optimizer.__next_solution(population[fitness.index(max(fitness))], population_size=100)
        new_fitness = optimizer.evaluate_fitness(new_solution)

        # Check if the new solution is better than the best solution found so far
        if new_fitness > best_fitness:
            best_solution = new_solution
            best_fitness = new_fitness

        # Update the population
        population = next_population

    # Print the best solution and its fitness
    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)