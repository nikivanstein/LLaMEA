import numpy as np
import random

class HybridEvoAlg:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.crossover_prob = 0.8  # New addition: Crossover probability

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            solution = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(solution)
        return population

    def fitness(self, solution, func):
        return func(*solution)

    def selection(self, population):
        fitnesses = [self.fitness(solution, func) for solution in population]
        fitnesses = np.array(fitnesses)
        fitness_min = np.min(fitnesses)
        fitness_max = np.max(fitnesses)
        selection_probabilities = (fitnesses - fitness_min) / (fitness_max - fitness_min)
        selected_indices = np.random.choice(len(population), size=self.population_size, p=selection_probabilities)
        return [population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_prob:  # New addition: Crossover probability
            child = [0.5 * (parent1[i] + parent2[i]) for i in range(self.dim)]
            return child
        else:
            return parent1

    def mutation(self, solution):
        for i in range(self.dim):
            if random.random() < 0.1:
                solution[i] += random.uniform(-1.0, 1.0)
                solution[i] = max(-5.0, min(5.0, solution[i]))
        return solution

    def local_search(self, solution):
        best_solution = solution
        for i in range(self.dim):
            for new_solution in [solution[:i] + [solution[i] + 1.0] + solution[i+1:],
                                solution[:i] + [solution[i] - 1.0] + solution[i+1:]]:
                fitness = self.fitness(new_solution, func)
                if fitness < self.fitness(best_solution, func):
                    best_solution = new_solution
        return best_solution

    def hybrid_evo_alg(self, func):
        for _ in range(self.budget):
            population = self.selection(self.population)
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                new_population.append(child)
            self.population = new_population
            best_solution = max(self.population, key=self.fitness)
            if self.fitness(best_solution, func) < self.best_fitness:
                self.best_solution = best_solution
                self.best_fitness = self.fitness(best_solution, func)
            if self.best_fitness < func(0):
                return self.best_solution
        return self.best_solution

# Example usage:
if __name__ == "__main__":
    # Define the functions to be optimized
    def func1(x):
        return sum(x**2)

    def func2(x):
        return sum([i**2 for i in x])

    def func3(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x])

    def func4(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x])

    def func5(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x])

    def func6(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x])

    def func7(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func8(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func9(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func10(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func11(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func12(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func13(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func14(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func15(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func16(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func17(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func18(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func19(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func20(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func21(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func22(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func23(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def func24(x):
        return sum([i**2 for i in x]) + sum([i**3 for i in x]) + sum([i**4 for i in x]) + sum([i**5 for i in x])

    def evaluate_fitness(solution, func):
        return func(*solution)

    def hybrid_evo_alg(func):
        algorithm = HybridEvoAlg(100, 10)
        for _ in range(1000):
            best_solution = algorithm.hybrid_evo_alg(func)
            print(f'Best solution: {best_solution}, Best fitness: {algorithm.fitness(best_solution, func)}')