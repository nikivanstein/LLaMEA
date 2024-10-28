import random
import numpy as np

class EvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim
        self.population_history = []

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
                self.population_history.append((individual, best_individual, fitness(individual)))

            if random.random() < 0.3:
                # Refine the solution by changing a random line of the selected individual
                for j in range(self.dim):
                    if random.random() < 0.5:
                        self.search_spaces[j] = (self.search_spaces[j][0] + random.uniform(-1, 1), self.search_spaces[j][1] + random.uniform(-1, 1))
                self.population[i] = np.array(self.population[i])
                self.fitness_scores[i] = fitness(individual)

        return self.population

    def mutate(self, individual):
        if random.random() < 0.01:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

# Example usage:
if __name__ == "__main__":
    # Create an instance of the EvolutionaryBlackBoxOptimizer class
    optimizer = EvolutionaryBlackBoxOptimizer(10, 2)

    # Define a black box function
    def func(x):
        return np.sin(x)

    # Optimize the function using the EvolutionaryBlackBoxOptimizer class
    optimized_individual = optimizer(func)

    # Print the optimized individual
    print("Optimized individual:", optimized_individual)