import random
import numpy as np

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class GeneticAlgorithm(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize the population with random individuals
        return [self.__call__(func) for func in range(self.population_size)]

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

    def selection(self):
        # Select the fittest individuals using tournament selection
        winners = []
        for _ in range(self.population_size):
            winner = max(set([self.__call__(func) for func in random.sample(self.search_space, 3)]), key=lambda x: x)
            winners.append(winner)
        return winners

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents to create a child
        child = [x for x in parent1 if x not in parent2]
        for x in parent2:
            if x not in child:
                child.append(x)
        return child

    def mutation(self, individual):
        # Perform mutation on an individual to introduce random changes
        mutated_individual = [x for x in individual if x not in self.search_space]
        for x in individual:
            if random.random() < 0.45:
                mutated_individual.append(x)
        return mutated_individual

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of an individual using the given function
        return self.__call__(individual)

    def fitness(self, individual):
        # Calculate the fitness of an individual
        return self.evaluate_fitness(individual)

    def run(self):
        # Run the genetic algorithm for a fixed number of generations
        for _ in range(100):
            # Select the fittest individuals
            winners = self.selection()

            # Create a new population by crossover and mutation
            self.population = [self.mutation(individual) for individual in winners]

            # Evaluate the fitness of the new population
            new_population = []
            for _ in range(self.population_size):
                individual = random.choice(self.population)
                new_population.append(self.evaluate_fitness(individual))
            self.population = new_population

            # Replace the old population with the new population
            self.population = self.population[:self.population_size]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization using Genetic Algorithm
# Code: 
# ```python
# Novel Metaheuristic Algorithm for Black Box Optimization using Genetic Algorithm
# ```
# ```python
# ```