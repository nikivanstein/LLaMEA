import numpy as np
import random

class EvolutionaryProbabilisticAdaptation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()
        self.fitness = self.evaluate_fitness()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(individual)
        return population

    def evaluate_fitness(self):
        fitness = []
        for individual in self.population:
            fitness.append(self(func(individual)))
        return fitness

    def selection(self):
        fitness = self.evaluate_fitness()
        probabilities = np.array(fitness) / np.sum(fitness)
        selected_indices = np.random.choice(len(fitness), size=self.population_size, p=probabilities, replace=False)
        selected_population = [self.population[i] for i in selected_indices]
        return selected_population

    def crossover(self, parent1, parent2):
        child = []
        for i in range(self.dim):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def mutation(self, individual):
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if random.random() < 0.1:
                mutated_individual[i] += random.uniform(-1.0, 1.0)
                if mutated_individual[i] < -5.0:
                    mutated_individual[i] = -5.0
                elif mutated_individual[i] > 5.0:
                    mutated_individual[i] = 5.0
        return mutated_individual

    def update_population(self):
        selected_population = self.selection()
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(selected_population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutation(child)
            new_population.append(child)
        self.population = new_population

    def __call__(self, func):
        for _ in range(self.budget):
            self.update_population()
            best_individual = max(self.population, key=lambda individual: self.func(individual))
            print(f"Best individual: {best_individual}, Fitness: {self.func(best_individual)}")
            if self.budget - _ <= 10:
                print("Converged!")

def func(individual):
    # Replace this with your black box function
    return np.sum([i**2 for i in individual])

# Usage
epa = EvolutionaryProbabilisticAdaptation(100, 5)
epa()