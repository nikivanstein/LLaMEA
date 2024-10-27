import numpy as np
import random

class PRES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.evolution_strategy = self.evolution_strategy_fn

    def evolution_strategy_fn(self, func):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(self.search_space[0], self.search_space[1], self.dim) for _ in range(self.population_size)]

        # Initialize the best point and its score
        best_point = population[0]
        best_score = func(best_point)

        # Evolve the population using the evolutionary strategy
        for _ in range(self.budget):
            # Evaluate the population
            scores = [func(point) for point in population]

            # Select the best points
            best_points = np.array(population)[np.argsort(scores)]

            # Crossover and mutate the selected points
            new_population = []
            for i in range(self.population_size):
                if random.random() < self.crossover_rate:
                    parent1, parent2 = best_points[i], best_points[(i+1)%self.population_size]
                    child = (parent1 + parent2) / 2
                    if random.random() < self.mutation_rate:
                        child += np.random.uniform(-self.search_space[1], self.search_space[1], self.dim)
                else:
                    child = best_points[i]
                new_population.append(child)

            # Replace the old population with the new one
            population = new_population

            # Update the best point and its score
            scores = [func(point) for point in population]
            best_point = population[np.argmax(scores)]
            best_score = scores[np.argmax(scores)]

        return best_point, best_score

    def __call__(self, func):
        return self.evolution_strategy_fn(func)

# Example usage:
if __name__ == "__main__":
    budget = 100
    dim = 10
    pres = PRES(budget, dim)
    for i in range(budget):
        best_point, best_score = pres(func)
        print(f"Function evaluation {i+1}: {best_score}")