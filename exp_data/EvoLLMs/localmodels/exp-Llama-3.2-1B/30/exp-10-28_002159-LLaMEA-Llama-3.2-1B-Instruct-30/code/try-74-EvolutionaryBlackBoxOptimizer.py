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
        if random.random() < 0.3:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

    def evolve(self):
        for _ in range(self.budget):
            new_population = self.population.copy()
            for _ in range(self.population_size):
                if random.random() < 0.5:
                    new_individual = self.mutate(new_population[_])
                else:
                    new_individual = self.population[_]
                new_population[_] = new_individual
            self.population = new_population
        return self.population

    def evaluate_bbob(self):
        best_individual = self.population[np.argmax(self.fitness_scores)]
        return best_individual

    def mutate_bbob(self, individual):
        if random.random() < 0.3:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate_bbobx(self):
        best_individual = self.evaluate_bbob()
        return best_individual

# Example usage:
optimizer = EvolutionaryBlackBoxOptimizer(100, 10)
best_individual = optimizer.evaluate_bbobx()
print("Best individual:", best_individual)

# Select the best individual and apply the mutation strategy
best_individual = optimizer.evaluate_bbobx()
optimizer.mutate_bbobx(best_individual)
print("Best individual after mutation:", best_individual)

# Update the population using the best individual
best_individual = optimizer.evaluate_bbobx()
optimizer.population = optimizer.evolve()
print("Updated population:", optimizer.population)