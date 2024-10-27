import numpy as np
import random

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm for solving black box optimization problems.

    The algorithm uses a combination of genetic algorithm and simulated annealing to find the optimal solution.
    """

    def __init__(self, budget, dim):
        """
        Initializes the optimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.generate_initial_population()
        self.crossover_probability = 0.5
        self.mutation_probability = 0.1

    def generate_initial_population(self):
        """
        Generates the initial population of individuals.

        Returns:
            list: A list of individuals, each represented as a list of floats.
        """
        return [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(self.population_size)]

    def __call__(self, func):
        """
        Optimizes a black box function using the optimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimal solution and the number of function evaluations used.
        """
        # Initialize the solution and the number of function evaluations
        solution = None
        evaluations = 0

        # Iterate over the range of possible solutions
        while evaluations < self.budget:
            # Generate a random solution within the search space
            solution = self.generate_solution()

            # Evaluate the black box function at the current solution
            evaluations += 1
            func(solution)

            # If the current solution is better than the previous best solution, update the solution
            if evaluations > 0 and evaluations < self.budget:
                if evaluations > 0:
                    # Calculate the probability of accepting the current solution
                    probability = np.exp((evaluations - evaluations) / self.budget)

                    # Accept the current solution with a probability less than 1
                    if np.random.rand() < probability:
                        solution = self.update_solution(solution)

                else:
                    # Update the solution with the best solution found so far
                    solution = None

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


    def generate_solution(self):
        """
        Generates a random solution within the search space.

        Returns:
            list: A list of floats representing the solution.
        """
        return [random.uniform(-5.0, 5.0) for _ in range(self.dim)]


    def update_solution(self, solution):
        """
        Updates the solution with a new individual.

        Args:
            solution (list): The new individual.

        Returns:
            list: The updated solution.
        """
        # Calculate the fitness of the current solution
        fitness = self.evaluate_fitness(solution)

        # Select the fittest individual
        fittest_individual = self.population[np.argmax(self.evaluate_fitness(self.population))]

        # Create a new individual by crossover and mutation
        offspring = []
        while len(offspring) < 2:
            parent1, parent2 = random.sample(self.population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            offspring.append(child)

        # Return the new individual
        return offspring


    def crossover(self, parent1, parent2):
        """
        Performs crossover between two individuals.

        Args:
            parent1 (list): The first individual.
            parent2 (list): The second individual.

        Returns:
            list: The child individual.
        """
        # Calculate the crossover point
        crossover_point = random.randint(1, self.dim - 1)

        # Create the child individual
        child = parent1[:crossover_point] + parent2[crossover_point:]

        return child


    def mutate(self, individual):
        """
        Mutates an individual.

        Args:
            individual (list): The individual to mutate.

        Returns:
            list: The mutated individual.
        """
        # Randomly swap two elements in the individual
        if random.random() < self.mutation_probability:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]

        return individual


    def evaluate_fitness(self, individual):
        """
        Evaluates the fitness of an individual.

        Args:
            individual (list): The individual to evaluate.

        Returns:
            float: The fitness of the individual.
        """
        # Calculate the fitness as the sum of the squares of the elements
        fitness = sum([x**2 for x in individual])

        return fitness


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Evolutionary optimization using Genetic Algorithm and Simulated Annealing
# Code: 