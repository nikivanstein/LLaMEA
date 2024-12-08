import numpy as np
import random
import math

class EvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitnesses = []

    def __call__(self, func, mutation_rate=0.1, crossover_rate=0.7, mutation_prob=0.3):
        """
        Optimize the black box function using evolutionary strategies.

        Args:
            func (function): The black box function to optimize.
            mutation_rate (float, optional): The rate at which to introduce genetic mutations. Defaults to 0.1.
            crossover_rate (float, optional): The rate at which to perform crossover operations. Defaults to 0.7.
            mutation_prob (float, optional): The probability of introducing genetic mutations. Defaults to 0.3.

        Returns:
            tuple: A tuple containing the optimized value of the function and its fitness.
        """
        # Initialize population using random initialization
        for _ in range(100):
            individual = np.random.rand(self.dim)
            self.population.append(individual)
            self.fitnesses.append(func(individual))

        # Run evolutionary algorithm
        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = self.select_parents()

            # Perform crossover and mutation operations
            offspring = []
            for _ in range(len(parents)):
                parent1, parent2 = random.sample(parents, 2)
                child = crossover(parent1, parent2, crossover_rate, mutation_prob)
                if random.random() < mutation_prob:
                    child = mutate(child, mutation_rate)
                offspring.append(child)

            # Replace parents with offspring
            parents = offspring

            # Evaluate fitness of offspring
            self.fitnesses = [func(individual) for individual in parents]

            # Select best individual
            best_individual = self.select_best()

            # Replace parents with best individual
            parents = [best_individual] + parents

            # Update population
            self.population = parents

        # Return optimized function and its fitness
        return func(best_individual), np.mean(self.fitnesses)

    def select_parents(self):
        """
        Select parents using tournament selection.

        Returns:
            list: A list of parent individuals.
        """
        winners = []
        for _ in range(self.budget):
            winner = np.random.choice(self.population, 1, replace=False)
            winners.append(winner)
        return winners

    def crossover(self, parent1, parent2, crossover_rate, mutation_prob):
        """
        Perform crossover operation between two parents.

        Args:
            parent1 (array): The first parent.
            parent2 (array): The second parent.
            crossover_rate (float): The rate at which to perform crossover operations.
            mutation_prob (float): The probability of introducing genetic mutations.

        Returns:
            array: The offspring.
        """
        if random.random() < crossover_rate:
            # Perform crossover operation
            child = np.concatenate((parent1[:len(parent1)//2], parent2[len(parent2)//2:]))
            return child
        else:
            return parent1

    def mutate(self, individual, mutation_rate):
        """
        Introduce genetic mutations into an individual.

        Args:
            individual (array): The individual.
            mutation_rate (float): The rate at which to introduce genetic mutations.

        Returns:
            array: The mutated individual.
        """
        if random.random() < mutation_rate:
            # Introduce genetic mutation
            return individual + np.random.rand(len(individual))
        else:
            return individual

    def select_best(self):
        """
        Select the best individual from the current population.

        Returns:
            array: The best individual.
        """
        return np.argmax(self.fitnesses)

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

optimizer = EvolutionaryOptimizer(100, 2)
optimized_func, fitness = optimizer(func, mutation_rate=0.2, crossover_rate=0.8, mutation_prob=0.4)
print(f"Optimized function: {optimized_func}, Fitness: {fitness}")