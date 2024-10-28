import random
import numpy as np

class AdaptiveEvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim
        self.search_space_deltas = np.zeros((self.dim, 2))

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_spaces[i][0], self.search_spaces[i][1]) for i in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

        for _ in range(self.budget):
            for i, individual in enumerate(self.population):
                fitness_scores[i] = fitness(individual)
                best_individual = self.population[np.argmax(fitness_scores)]
                new_individual = [random.uniform(self.search_spaces[j][0], self.search_spaces[j][1]) for j in range(self.dim)]
                new_individual = np.array(new_individual)
                if fitness(individual) > fitness(best_individual):
                    self.population[i] = new_individual
                    self.fitness_scores[i] = fitness(individual)

            # Update search space deltas based on fitness scores
            deltas = np.zeros((self.dim, 2))
            for i, individual in enumerate(self.population):
                delta = self.fitness_scores[i] / fitness_scores[np.argmax(fitness_scores)]
                deltas[i, 0] = random.uniform(-1, 1)
                deltas[i, 1] = random.uniform(-1, 1)
            self.search_space_deltas = np.vstack((self.search_space_deltas, deltas))

            # Select new population with updated search space deltas
            new_population = self.select_population()

            # Evaluate new population
            new_population = np.array(new_population)
            self.population = new_population

        return self.population

    def select_population(self):
        # Select new population based on search space deltas
        new_population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_spaces[j][0], self.search_spaces[j][1]) for j in range(self.dim)]
            if random.random() < 0.3:
                individual = self.search_space_deltas[np.argmax(self.search_space_deltas, axis=0)].reshape(-1, 1)
            new_population.append(individual)
        return np.array(new_population)

    def mutate(self, individual):
        if random.random() < 0.01:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

# Example usage
def black_box_function(x):
    return np.sum(x**2)

optimizer = AdaptiveEvolutionaryBlackBoxOptimizer(budget=1000, dim=2)
best_individual = optimizer.population[np.argmax(optimizer.fitness_scores)]
best_function_value = black_box_function(best_individual)

# Print the best solution
print("Best solution:", best_individual)
print("Best function value:", best_function_value)