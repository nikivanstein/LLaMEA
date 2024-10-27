import random
import numpy as np

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
        # Define the mutation strategy
        def mutate(individual):
            # Randomly select a point in the search space
            new_point = self.search_space[np.random.randint(0, self.dim)]

            # Check if the new point is within the search space
            if not (self.search_space >= new_point >= -self.search_space):
                return individual

            # Mutate the individual by swapping two random points
            return individual[:self.dim//2] + [new_point[self.dim//2]] + individual[self.dim//2+1:]

        # Define the crossover strategy
        def crossover(parent1, parent2):
            # Select a random crossover point
            crossover_point = np.random.randint(0, self.dim)

            # Create a new child by combining the two parents
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

            # Check if the child is within the search space
            if not (self.search_space >= child >= -self.search_space):
                return child

            # Return the child
            return child

        # Define the selection strategy
        def select(parent):
            # Use tournament selection
            tournament_size = 3
            tournament_results = np.random.randint(0, self.budget, size=tournament_size)

            # Select the best individual based on tournament results
            return np.argmax(tournament_results)

        # Initialize the population with random individuals
        population = [mutate(np.random.choice(self.search_space, self.dim)) for _ in range(100)]

        # Evolve the population over many generations
        for _ in range(1000):
            # Evaluate the fitness of each individual
            fitnesses = [self.func(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = [individual for _, individual in sorted(zip(fitnesses, population), reverse=True)]

            # Crossover and mutate the fittest individuals
            new_population = []
            for i in range(0, len(fittest_individuals), 2):
                parent1, parent2 = fittest_individuals[i], fittest_individuals[i+1]
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)

            # Replace the old population with the new one
            population = new_population

        # Return the best individual in the final population
        return population[0]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 