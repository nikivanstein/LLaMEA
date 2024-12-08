import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        population_size = 100
        while True:
            # Select parents using tournament selection
            parents = self.select_parents(population_size)

            # Create offspring using crossover and mutation
            offspring = self.crossover(parents)
            # Mutate offspring with probability 0.05
            offspring = self.mutate(offspring, self.budget)

            # Evaluate fitness of offspring
            fitnesses = [self.evaluate_fitness(offspring[i]) for i in range(population_size)]
            # Select best individual
            best_individual = np.argmax(fitnesses)

            # Replace worst individual with best individual
            self.search_space[np.argmax(fitnesses) + 1:] = offspring
            self.search_space = np.delete(self.search_space, 0, axis=0)

            # Check if we have reached the budget
            if np.sum(np.abs(self.search_space)) < self.budget / 2:
                break

    def select_parents(self, population_size):
        # Select parents using tournament selection
        parents = []
        for _ in range(population_size):
            tournament_size = random.randint(1, self.search_space.shape[0])
            tournament_indices = np.random.choice(self.search_space.shape[0], tournament_size, replace=False)
            winner_indices = np.argsort(self.search_space[tournament_indices])[:tournament_size]
            winner_indices = winner_indices[np.argsort(self.search_space[winner_indices])]
            winner_individual = self.search_space[winner_indices]
            parents.append(winner_individual)
        return np.array(parents)

    def crossover(self, parents):
        # Create offspring using crossover
        offspring = parents.copy()
        for i in range(len(offspring)):
            if np.random.rand() < 0.5:
                offspring[i] = np.concatenate((parents[i], parents[np.random.randint(0, len(parents))]))
            else:
                offspring[i] = np.concatenate((parents[i], parents[np.random.randint(0, len(parents) - 1)]))
        return offspring

    def mutate(self, offspring, budget):
        # Mutate offspring with probability 0.05
        mutated_offspring = []
        for individual in offspring:
            if np.random.rand() < 0.05:
                # Swap two random elements in the individual
                idx1 = random.randint(0, len(individual) - 1)
                idx2 = random.randint(0, len(individual) - 1)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
                mutated_offspring.append(individual)
            else:
                # Shuffle the individual
                mutated_offspring.append(individual.copy())
        return mutated_offspring

    def evaluate_fitness(self, individual):
        # Evaluate fitness of individual using the original function
        return self.func(individual)