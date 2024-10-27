import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.01
        self.tournament_size = 2

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Initialize a random population of individuals
            population = self.generate_population(self.population_size)

            # Evaluate the population
            fitnesses = [self.evaluate_fitness(individual, func) for individual in population]

            # Select the fittest individuals for tournament
            tournament = random.sample(population, self.tournament_size)
            tournament_fitnesses = [self.evaluate_fitness(individual, func) for individual in tournament]

            # Calculate the tournament selection probability
            selection_probabilities = [fitness / np.sum(fitnesses) for fitness in fitnesses]
            tournament_selection_probabilities = [np.random.rand() for _ in range(self.population_size)]
            tournament_selection_probabilities = [p for p in tournament_selection_probabilities if p >= selection_probabilities[i]]

            # Perform tournaments
            tournament_results = []
            for i in range(self.tournament_size):
                # Select two random individuals for the tournament
                individual1, individual2 = random.sample(tournament, 2)

                # Evaluate the two individuals in the tournament
                tournament_fitnesses = [self.evaluate_fitness(individual1, func), self.evaluate_fitness(individual2, func)]

                # Calculate the tournament selection probability
                selection_probabilities = [fitness / np.sum(fitnesses) for fitness in tournament_fitnesses]
                tournament_selection_probabilities = [np.random.rand() for _ in range(self.population_size)]
                tournament_selection_probabilities = [p for p in tournament_selection_probabilities if p >= selection_probabilities[i]]

                # Perform the tournament
                tournament_result = self.tournament_selection(individual1, individual2, func, tournament_fitnesses, tournament_selection_probabilities)

                # Add the tournament result to the list
                tournament_results.append(tournament_result)

            # Calculate the average fitness of the tournament
            tournament_average_fitness = np.mean([result[1] for result in tournament_results])

            # Update the population
            population = [individual for individual, result in zip(population, tournament_results) if result[0] == tournament_average_fitness]

            # Update the fitness of the best individual
            best_individual = max(population, key=self.evaluate_fitness)
            best_individual_fitness = self.evaluate_fitness(best_individual, func)

            # Update the population size
            self.population_size *= 0.99

            # Update the budget
            self.budget -= 1

            # Update the search space
            self.search_space = (-5.0 + 0.01 * random.uniform(-5.0, 5.0), 5.0 + 0.01 * random.uniform(-5.0, 5.0))

            # Update the fitness of the best individual
            best_individual_fitness = self.evaluate_fitness(best_individual, func)
            self.func_evaluations += 1

            # Update the best individual
            if best_individual_fitness > self.func_evaluations / self.budget:
                best_individual = best_individual_fitness, self.search_space[0], self.search_space[1]
            else:
                best_individual = self.search_space[0], self.search_space[1]

            # Return the best individual
            return best_individual

    def generate_population(self, size):
        population = []
        for _ in range(size):
            individual = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            population.append(individual)
        return population

    def evaluate_fitness(self, individual, func):
        func_value = func(individual)
        return func_value

    def tournament_selection(self, individual1, individual2, func, fitnesses, selection_probabilities):
        # Select the two individuals with the highest fitness
        selected_individual = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        return individual1, individual2