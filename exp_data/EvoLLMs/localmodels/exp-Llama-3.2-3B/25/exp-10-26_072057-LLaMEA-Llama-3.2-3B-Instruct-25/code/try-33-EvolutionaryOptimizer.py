import numpy as np
import random

class EvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.swarm_size = 10

    def __call__(self, func):
        # Initialize population with random solutions
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        for _ in range(self.budget):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Select fittest individuals
            fittest = population[np.argsort(fitness)]

            # Generate new offspring
            offspring = []
            for _ in range(self.population_size):
                # Select parents
                parent1, parent2 = random.sample(fittest, 2)

                # Crossover
                child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))

                # Mutate
                if random.random() < self.mutation_rate:
                    child += np.random.uniform(-0.1, 0.1, self.dim)

                offspring.append(child)

            # Replace least fit individuals with new offspring
            population = np.array(offspring)

        # Select the best solution
        best_solution = population[np.argmin(fitness)]

        # Refine the best solution by changing 25% of its lines
        refined_solution = best_solution.copy()
        refined_solution = self.refine_solution(refined_solution, 0.25)

        return refined_solution

    def refine_solution(self, solution, probability):
        # Get the indices of the solution
        indices = np.where(solution!= solution[0])

        # Randomly select indices to change
        indices_to_change = np.random.choice(indices[0], size=int(len(indices[0]) * probability), replace=False)

        # Change the values at the selected indices
        refined_solution = solution.copy()
        refined_solution[indices_to_change] = np.random.uniform(-5.0, 5.0, size=len(indices[0]))

        return refined_solution

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = EvolutionaryOptimizer(budget=100, dim=5)
best_solution = optimizer(func)
print("Best solution:", best_solution)