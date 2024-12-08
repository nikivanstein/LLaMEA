import numpy as np
import random

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def refine_strategy(self, func):
        # Refine the search space using genetic algorithm
        population = self.generate_population()
        for _ in range(10):  # Run 10 generations
            # Select parents using tournament selection
            parents = self.select_parents(population)
            # Crossover (reproduce) offspring using uniform crossover
            offspring = self.crossover(parents)
            # Mutate offspring using bit-flip mutation
            offspring = self.mutate(offspring)
            # Replace the old population with the new one
            population = self.replace_population(population, offspring)

    def generate_population(self):
        # Generate a population of random solutions
        population = []
        for _ in range(100):
            solution = self.search_space + np.random.uniform(-1, 1, self.dim)
            population.append(solution)
        return population

    def select_parents(self, population):
        # Select parents using tournament selection
        tournament_size = 3
        tournament_results = []
        for _ in range(100):
            parent1, parent2, parent3 = random.sample(population, tournament_size)
            results = [np.abs(x - y) for x, y in zip(parent1, parent2) if np.abs(x - y) < 1e-6]
            tournament_results.append(results[0])
        tournament_results = np.array(tournament_results).reshape(-1, tournament_size)
        return tournament_results

    def crossover(self, parents):
        # Crossover (reproduce) offspring using uniform crossover
        offspring = parents.copy()
        for i in range(len(parents) // 2):
            if random.random() < 0.5:  # 50% chance of crossover
                parent1, parent2 = parents[i], parents[-i - 1]
                idx = random.randint(0, len(parent1) - 1)
                offspring[idx] = parent2[idx]
        return offspring

    def mutate(self, offspring):
        # Mutate offspring using bit-flip mutation
        mutated_offspring = []
        for solution in offspring:
            mutated_solution = solution.copy()
            for i in range(len(mutated_solution)):
                if random.random() < 0.1:  # 10% chance of mutation
                    mutated_solution[i] ^= 1
            mutated_offspring.append(mutated_solution)
        return mutated_offspring

    def replace_population(self, population, offspring):
        # Replace the old population with the new one
        population[:] = offspring
        return population

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

dabu.refine_strategy(test_function)
print(dabu(test_function))  # prints a refined value between -10 and 10