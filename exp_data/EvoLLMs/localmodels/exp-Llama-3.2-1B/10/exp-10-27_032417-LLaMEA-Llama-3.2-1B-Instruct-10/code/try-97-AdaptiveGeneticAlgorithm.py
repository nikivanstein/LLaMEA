import numpy as np
import random

class AdaptiveGeneticAlgorithm:
    """
    An adaptive genetic algorithm for solving black box optimization problems.
    """

    def __init__(self, budget, dim, adaptive_coefficient=0.1):
        """
        Initializes the adaptive genetic algorithm with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
            adaptive_coefficient (float, optional): The coefficient for adaptive selection. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.adaptive_coefficient = adaptive_coefficient

    def __call__(self, func):
        """
        Optimizes a black box function using the adaptive genetic algorithm.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimal solution and the number of function evaluations used.
        """
        # Initialize the solution and the number of function evaluations
        solution = None
        evaluations = 0

        # Initialize the population with random solutions
        population = self.generate_population(func, self.budget)

        # Iterate until the population is converged or the budget is reached
        while evaluations < self.budget and population:
            # Select parents using a combination of genetic and adaptive selection
            parents = self.select_parents(population)

            # Crossover and mutation to generate offspring
            offspring = self.crossover_and_mutate(parents)

            # Evaluate the fitness of each individual
            fitness = [self.evaluate_fitness(individual, func) for individual in offspring]

            # Sort the offspring by fitness
            offspring.sort(key=lambda x: x[1], reverse=True)

            # Select the fittest individuals to reproduce
            selected_offspring = offspring[:int(self.adaptive_coefficient * len(offspring))]

            # Replace the least fit individuals with the selected ones
            population = selected_offspring + offspring[len(selected_offspring):]

            # Update the solution with the fittest individual
            solution = selected_offspring[-1]

            # Evaluate the fitness of the new solution
            evaluations += 1
            fitness = self.evaluate_fitness(solution, func)

            # If the new solution is better than the previous best solution, update the solution
            if evaluations > 0 and evaluations < self.budget:
                if evaluations > 0:
                    # Calculate the probability of accepting the new solution
                    probability = np.exp((evaluations - evaluations) / self.budget)

                    # Accept the new solution with a probability less than 1
                    if np.random.rand() < probability:
                        solution = solution
                else:
                    # Update the solution with the best solution found so far
                    solution = None

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


    def generate_population(self, func, budget):
        """
        Generates a population of random solutions.

        Args:
            func (function): The black box function to optimize.
            budget (int): The maximum number of function evaluations allowed.

        Returns:
            list: A list of random solutions.
        """
        population = []
        for _ in range(budget):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population


    def select_parents(self, population):
        """
        Selects parents using a combination of genetic and adaptive selection.

        Args:
            population (list): A list of random solutions.

        Returns:
            list: A list of selected parents.
        """
        # Select parents using genetic selection
        parents = random.choices(population, k=10)
        return parents


    def crossover_and_mutate(self, parents):
        """
        Crossover and mutation to generate offspring.

        Args:
            parents (list): A list of selected parents.

        Returns:
            list: A list of offspring.
        """
        offspring = []
        for _ in range(10):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = (parent1 + parent2) / 2
            offspring.append((child, parent1[1] + parent2[1]))
        return offspring


    def evaluate_fitness(self, individual, func):
        """
        Evaluates the fitness of an individual.

        Args:
            individual (tuple): The individual to evaluate.
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the fitness and the individual's index.
        """
        fitness = func(individual)
        return fitness, individual


# Example usage:
def func(x):
    return x**2 + 2*x + 1

adaptiveGA = AdaptiveGeneticAlgorithm(100, 10)
optimal_solution, num_evaluations = adaptiveGA(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)