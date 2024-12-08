import numpy as np
import random

class EvolutionaryGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.population = self.initialize_population()
        self.best_solution = self.evaluate_best_solution()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            solution = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(solution)
        return population

    def evaluate_best_solution(self):
        best_solution = min(self.population, key=lambda x: self.func(x))
        return best_solution

    def __call__(self, func):
        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = self.select_parents()

            # Crossover (recombination) with probability 0.7
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                child = self.recombine(parent1, parent2)
                offspring.append(child)

            # Mutate with probability 0.3
            mutated_offspring = []
            for child in offspring:
                if random.random() < 0.3:
                    mutated_child = self.mutate(child)
                    mutated_offspring.append(mutated_child)
                else:
                    mutated_offspring.append(child)

            # Replace the worst solution with the new offspring
            self.population = mutated_offspring
            self.best_solution = self.evaluate_best_solution()

            # Evaluate the function at the new best solution
            self.func(self.best_solution)

    def select_parents(self):
        parents = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, 5)
            winner = min(tournament, key=lambda x: self.func(x))
            parents.append(winner)
        return parents

    def recombine(self, parent1, parent2):
        child = [0.5 * (parent1[i] + parent2[i]) for i in range(self.dim)]
        return child

    def mutate(self, child):
        mutated_child = child.copy()
        for i in range(self.dim):
            if random.random() < 0.3:
                mutated_child[i] += random.uniform(-1.0, 1.0)
                mutated_child[i] = max(-5.0, min(5.0, mutated_child[i]))
        return mutated_child

    def func(self, solution):
        # Evaluate the function at the given solution
        # Replace this with your actual function
        return np.sum([solution[i]**2 for i in range(self.dim)])

# Example usage
eg = EvolutionaryGradient(budget=100, dim=10)
eg.func(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))