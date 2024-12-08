import random
import numpy as np

class EvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim, mutation_rate=0.01, adaptive Mutation=False):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim
        self.mutation_rate = mutation_rate
        self.adaptiveMutation = adaptiveMutation

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

        return self.population

    def mutate(self, individual):
        if random.random() < self.mutation_rate and not self.adaptiveMutation:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

    def update(self):
        if not self.population:
            return self

        # Select the best individual
        best_individual = self.population[np.argmax(self.fitness_scores)]

        # Select a random individual to refine the best individual
        refined_individual = random.choice([individual for individual in self.population if individual!= best_individual])

        # Perform adaptive mutation if required
        if self.adaptiveMutation:
            refined_individual = self.adaptiveMutation(refined_individual)

        # Update the best individual
        best_individual = refined_individual

        # Replace the best individual in the population
        self.population[np.argmax(self.fitness_scores)] = best_individual

        return self

    def __str__(self):
        return "EvolutionaryBlackBoxOptimizer: Evolutionary Black Box Optimization"