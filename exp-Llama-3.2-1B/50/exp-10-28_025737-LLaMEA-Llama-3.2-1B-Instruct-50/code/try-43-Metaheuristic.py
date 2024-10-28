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

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Initialize population with random individuals
        population = [self.__init__(self.budget, self.dim) for _ in range(100)]

        # Evolve population for 100 generations
        for _ in range(100):
            # Evaluate population fitness
            fitnesses = [individual.func(self.search_space) for individual in population]

            # Select parents using tournament selection
            parents = random.sample(population, min(self.budget, len(population) // 2))

            # Create offspring using crossover and mutation
            offspring = []
            for _ in range(self.budget):
                parent1, parent2 = random.sample(parents, 2)
                child = parent1.func(self.search_space)
                if random.random() < 0.45:
                    child = random.uniform(parent1.search_space)
                offspring.append(child)

            # Replace worst individuals with offspring
            population = [individual for individual in population if individual.func(self.search_space) < fitnesses[0]]
            population += offspring

        # Return best individual
        return population[0]

# Evaluate the algorithm on the BBOB test suite
bboo = BBOB()
results = []
for name, description, score in bboo.evaluate():
    algorithm = NovelMetaheuristicAlgorithm(100, 10)
    best_individual = algorithm(bboo.func)
    results.append((name, description, score, best_individual.func(self.search_space)))