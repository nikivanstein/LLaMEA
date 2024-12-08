import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

class MetaLearningBasedDiversityDrivenEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.population = self.initialize_population()
        self.model = self.initialize_model()

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def initialize_model(self):
        X_train, X_test, y_train, y_test = train_test_split(np.random.uniform(-5.0, 5.0, (self.population_size, self.dim)), np.random.uniform(-5.0, 5.0, self.population_size), test_size=0.2, random_state=42)
        model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        return model

    def evaluate(self, func):
        fitnesses = func(self.population)
        self.population = self.select_parents(fitnesses)
        self.population = self.crossover(self.population)
        self.population = self.mutate(self.population)
        predictions = self.model.predict(self.population)
        return np.array([np.mean(prediction) for prediction in predictions])

    def select_parents(self, fitnesses):
        fitnesses = np.array(fitnesses)
        parents = np.array([self.population[np.argsort(fitnesses)[:int(self.population_size/2)]]])
        return parents

    def crossover(self, population):
        offspring = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            if random.random() < self.crossover_probability:
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                offspring[i] = (parent1 + parent2) / 2
        return offspring

    def mutate(self, population):
        mutated_population = np.copy(population)
        for i in range(self.population_size):
            if random.random() < self.mutation_probability:
                mutated_population[i] += np.random.uniform(-1.0, 1.0, self.dim)
        return mutated_population

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.population, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

ddea = MetaLearningBasedDiversityDrivenEvolutionaryAlgorithm(budget=100, dim=10)
optimal_solution = ddea(func)
print(optimal_solution)