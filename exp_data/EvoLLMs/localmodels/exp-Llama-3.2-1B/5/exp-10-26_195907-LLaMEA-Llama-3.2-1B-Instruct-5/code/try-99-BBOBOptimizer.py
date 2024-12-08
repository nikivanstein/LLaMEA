import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        # Initialize population with random individuals
        population = [self.generate_individual() for _ in range(1000)]

        while True:
            # Evaluate fitness of each individual
            fitnesses = [self.evaluate_fitness(individual, self.func) for individual in population]

            # Select parents using tournament selection
            parents = self.select_parents(population, fitnesses)

            # Apply simulated annealing to refine strategy
            new_individuals = self.simulated_annealing(parents, fitnesses, self.budget, self.dim)

            # Replace old population with new ones
            population = new_individuals

            # Check for convergence
            if np.all(population == [self.generate_individual() for _ in range(1000)]):
                break

        # Return the fittest individual
        return population[np.argmax(fitnesses)]

    def generate_individual(self):
        # Create an individual by sampling from the search space
        return self.search_space[np.random.randint(0, self.search_space.shape[0])]

    def select_parents(self, population, fitnesses):
        # Select parents using tournament selection
        parents = []
        for _ in range(self.budget):
            individual = random.choice(population)
            tournament_size = 5
            winners = np.array([self.evaluate_fitness(individual, self.func) for _ in range(tournament_size)])
            winner_index = np.argmax(winners)
            parents.append(individual[winner_index])
        return parents

    def simulated_annealing(self, parents, fitnesses, budget, dim):
        # Apply simulated annealing to refine strategy
        new_individuals = []
        temperature = 1.0
        for _ in range(budget):
            individual = random.choice(parents)
            delta_fitness = self.evaluate_fitness(individual, self.func) - self.evaluate_fitness(individual, self.func)
            if delta_fitness > 0:
                new_individual = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                new_individual = np.vstack((new_individual, individual))
                new_individuals.append(new_individual)
            elif random.random() < np.exp(-delta_fitness / temperature):
                new_individual = individual
            else:
                new_individual = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                new_individuals.append(new_individual)
            self.search_space = np.vstack((self.search_space, new_individual))
            self.search_space = np.delete(self.search_space, 0, axis=0)
            temperature *= 0.99
        return new_individuals