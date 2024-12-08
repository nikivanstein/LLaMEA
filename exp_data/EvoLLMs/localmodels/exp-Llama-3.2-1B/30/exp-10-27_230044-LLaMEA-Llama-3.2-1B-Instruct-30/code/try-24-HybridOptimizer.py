import numpy as np
import random
import math

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.generate_initial_population()
        self.population_history = []
        self.worst_individual_history = []

    def generate_initial_population(self):
        # Initialize population with random individuals
        return [(np.random.rand(self.dim) - 0.5, np.random.rand(self.dim) - 0.5) for _ in range(self.population_size)]

    def evaluate_fitness(self, individual, func):
        # Optimize the function using the individual
        return func(individual)

    def __call__(self, func):
        """
        Optimize the black box function using the hybrid algorithm.

        Args:
            func (function): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize population
        self.population = self.generate_initial_population()

        # Evaluate fitness for each individual
        fitnesses = [self.evaluate_fitness(individual, func) for individual in self.population]

        # Select the best individual
        selected_individual = self.select_best_individual(fitnesses)

        # Mutate the selected individual
        mutated_individual = self.mutate(selected_individual)

        # Evaluate fitness for the mutated individual
        fitness = self.evaluate_fitness(mutated_individual, func)

        # Update the population
        self.population = [mutated_individual] + [selected_individual] * (self.population_size - len(self.population))

        # Update the population history
        self.population_history.append(self.population[-1])
        self.worst_individual_history.append(selected_individual)

        # Check if the optimization is successful
        if fitness > self.budget * 0.3:
            return fitness
        else:
            # Refine the strategy
            if self.population[-1][0] == self.population_history[-1][0] == self.population_history[-2][0]:
                # Increase the population size
                self.population_size *= 1.1
                # Refine the mutation strategy
                mutated_individual = self.mutate(self.population_history[-1])
            else:
                # Decrease the population size
                self.population_size /= 1.1
                # Refine the mutation strategy
                mutated_individual = self.mutate(selected_individual)

            # Evaluate fitness for the mutated individual
            fitness = self.evaluate_fitness(mutated_individual, func)

            # Update the population
            self.population = [mutated_individual] + [selected_individual] * (self.population_size - len(self.population))

            # Update the population history
            self.population_history.append(self.population[-1])
            self.worst_individual_history.append(selected_individual)

            # Check if the optimization is successful
            if fitness > self.budget * 0.3:
                return fitness
            else:
                # Refine the strategy
                if self.population[-1][0] == self.population_history[-1][0] == self.population_history[-2][0]:
                    # Increase the population size
                    self.population_size *= 1.1
                    # Refine the mutation strategy
                    mutated_individual = self.mutate(self.population_history[-1])
                else:
                    # Decrease the population size
                    self.population_size /= 1.1
                    # Refine the mutation strategy
                    mutated_individual = self.mutate(selected_individual)

                # Evaluate fitness for the mutated individual
                fitness = self.evaluate_fitness(mutated_individual, func)

                # Update the population
                self.population = [mutated_individual] + [selected_individual] * (self.population_size - len(self.population))

                # Update the population history
                self.population_history.append(self.population[-1])
                self.worst_individual_history.append(selected_individual)

                # Check if the optimization is successful
                if fitness > self.budget * 0.3:
                    return fitness
                else:
                    # Refine the strategy
                    if self.population[-1][0] == self.population_history[-1][0] == self.population_history[-2][0]:
                        # Increase the population size
                        self.population_size *= 1.1
                        # Refine the mutation strategy
                        mutated_individual = self.mutate(self.population_history[-1])
                    else:
                        # Decrease the population size
                        self.population_size /= 1.1
                        # Refine the mutation strategy
                        mutated_individual = self.mutate(selected_individual)

                    # Evaluate fitness for the mutated individual
                    fitness = self.evaluate_fitness(mutated_individual, func)

                    # Update the population
                    self.population = [mutated_individual] + [selected_individual] * (self.population_size - len(self.population))

                    # Update the population history
                    self.population_history.append(self.population[-1])
                    self.worst_individual_history.append(selected_individual)

                    # Check if the optimization is successful
                    if fitness > self.budget * 0.3:
                        return fitness
                    else:
                        # Refine the strategy
                        if self.population[-1][0] == self.population_history[-1][0] == self.population_history[-2][0]:
                            # Increase the population size
                            self.population_size *= 1.1
                            # Refine the mutation strategy
                            mutated_individual = self.mutate(self.population_history[-1])
                        else:
                            # Decrease the population size
                            self.population_size /= 1.1
                            # Refine the mutation strategy
                            mutated_individual = self.mutate(selected_individual)

    def mutate(self, individual):
        # Randomly mutate the individual
        mutated_individual = list(individual)
        for i in range(len(individual)):
            if random.random() < 0.5:
                mutated_individual[i] += random.uniform(-1, 1)
        return tuple(mutated_individual)

    def select_best_individual(self, fitnesses):
        # Select the best individual based on the fitness
        return fitnesses.index(max(fitnesses))

# Description: Hybrid Metaheuristic Algorithm for Black Box Optimization
# Code: 