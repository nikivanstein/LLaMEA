import random
import numpy as np

class EvolutionaryHyperParameterTuning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self._initialize_population()
        self.fitness_history = []

    def _initialize_population(self):
        population = []
        for _ in range(self.budget):
            individual = {}
            for i in range(self.dim):
                individual[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
            population.append(individual)
        return population

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_population(func)

    def _evaluate_and_mutate_population(self, func):
        fitnesses = [func(individual) for individual in self.population]
        self.fitness_history.append(fitnesses)
        if min(fitnesses) == max(fitnesses):
            return  # termination condition

        for individual in self.population:
            if random.random() < 0.35:  # mutation probability
                for i in range(self.dim):
                    if random.random() < 0.5:
                        individual[i]['value'] += random.uniform(-1.0, 1.0)
                        if individual[i]['value'] < individual[i]['lower']:
                            individual[i]['value'] = individual[i]['lower']
                        elif individual[i]['value'] > individual[i]['upper']:
                            individual[i]['value'] = individual[i]['upper']

        if random.random() < 0.35:  # crossover probability
            new_population = []
            for _ in range(len(self.population)):
                parent1, parent2 = random.sample(self.population, 2)
                child = {}
                for i in range(self.dim):
                    if random.random() < 0.5:
                        child[i]['value'] = parent1[i]['value']
                    else:
                        child[i]['value'] = parent2[i]['value']
                new_population.append(child)
            self.population = new_population

    def get_population(self):
        return self.population

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = EvolutionaryHyperParameterTuning(budget, dim)
evolution()
population = evolution.get_population()
print(population)