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

    def _mutation(self, individual, mutation_rate):
        """
        Perform a mutation on the individual.

        Args:
            individual (list): The individual to mutate.
            mutation_rate (float): The probability of mutation.

        Returns:
            list: The mutated individual.
        """
        # Generate a random mutation point
        mutation_point = np.random.randint(0, self.dim)

        # Swap the mutation point with a random point in the search space
        mutated_individual = copy.deepcopy(individual)
        mutated_individual[mutation_point], mutated_individual[mutation_point + self.dim] = mutated_individual[mutation_point + self.dim], mutated_individual[mutation_point]

        # If the mutation rate is greater than 0, apply a mutation
        if random.random() < mutation_rate:
            mutated_individual[mutation_point], mutated_individual[mutation_point + self.dim] = mutated_individual[mutation_point + self.dim], mutated_individual[mutation_point]

        return mutated_individual

    def _crossover(self, parents):
        """
        Perform crossover on two parents.

        Args:
            parents (list): The two parents to crossover.

        Returns:
            list: The offspring.
        """
        # Select a random crossover point
        crossover_point = np.random.randint(0, self.dim)

        # Create the offspring by combining the two parents
        offspring = [parents[0][i] for i in range(crossover_point)] + [parents[1][i] for i in range(crossover_point, self.dim)]

        return offspring

    def _selection(self, population):
        """
        Select the best individual from the population.

        Args:
            population (list): The population to select from.

        Returns:
            list: The selected individual.
        """
        # Select the top half of the population with a probability of selection
        selected_individual = random.choices(population, weights=[1 / len(population) for _ in range(len(population))], k=self.budget // 2)[0]

        return selected_individual

    def _fitness(self, individual):
        """
        Evaluate the fitness of an individual.

        Args:
            individual (list): The individual to evaluate.

        Returns:
            float: The fitness of the individual.
        """
        # Evaluate the function at the individual
        fitness = self.__call__(individual)

        # Return the fitness
        return fitness

    def optimize(self, func, population_size, mutation_rate, crossover_rate, selection_rate):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            population_size (int): The size of the population.
            mutation_rate (float): The probability of mutation.
            crossover_rate (float): The probability of crossover.
            selection_rate (float): The probability of selection.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the population
        population = [self._selection([func(np.random.rand(self.dim)) for _ in range(population_size)])]

        # Perform the specified number of iterations
        for _ in range(100):
            # Select the top half of the population with a probability of selection
            selected_population = self._selection(population)

            # Perform crossover on the selected population
            offspring = self._crossover(selected_population)

            # Perform mutation on the offspring
            mutated_offspring = self._mutation(offspring, mutation_rate)

            # Select the top half of the mutated offspring with a probability of selection
            selected_offspring = self._selection(mutated_offspring)

            # Replace the population with the selected offspring
            population = selected_offspring

        # Return the optimized value
        return self._fitness(population[0])