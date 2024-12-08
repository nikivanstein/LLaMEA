import numpy as np
import random
import time

class HybridEvoAlg:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.start_time = time.time()

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
        child = [0.5 * (parent1[i] + parent2[i]) for i in range(self.dim)]
        return child

    def mutation(self, solution):
        for i in range(self.dim):
            if random.random() < 0.1:
                solution[i] += random.uniform(-1.0, 1.0)
                solution[i] = max(-5.0, min(5.0, solution[i]))
        return solution

    def local_search(self, solution, func):
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

    def evaluate(self, func):
        current_time = time.time()
        print(f"Time elapsed: {current_time - self.start_time} seconds")
        self.hybrid_evo_alg(func)
        return self.best_solution

# Example usage
if __name__ == "__main__":
    func = lambda x: sum(x**2)
    algorithm = HybridEvoAlg(100, 10)
    solution = algorithm.evaluate(func)
    print(f"Best solution: {solution}")