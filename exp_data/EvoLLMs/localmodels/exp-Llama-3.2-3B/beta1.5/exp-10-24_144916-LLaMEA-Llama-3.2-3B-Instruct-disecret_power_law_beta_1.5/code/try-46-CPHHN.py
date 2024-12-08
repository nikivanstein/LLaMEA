import numpy as np
import random
import operator

class CPHHN:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()
        self.hypernetwork = self.initialize_hypernetwork()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def initialize_hypernetwork(self):
        hypernetwork = []
        for _ in range(self.population_size):
            hyperparameters = {
                'crossover_probability': np.random.uniform(0.0, 1.0),
               'mutation_probability': np.random.uniform(0.0, 1.0),
                'elitism_rate': np.random.uniform(0.0, 1.0)
            }
            hypernetwork.append(hyperparameters)
        return hypernetwork

    def evaluate(self, func):
        evaluations = []
        for individual in self.population:
            evaluation = func(individual)
            evaluations.append(evaluation)
        return evaluations

    def selection(self, evaluations):
        sorted_indices = np.argsort(evaluations)
        selected_indices = sorted_indices[:int(self.population_size * 0.2)]
        selected_individuals = [self.population[i] for i in selected_indices]
        return selected_individuals

    def probabilistic_crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dim)
        child = (parent1[:crossover_point] + parent2[crossover_point:]) if np.random.rand() < self.hypernetwork[0]['crossover_probability'] else parent1
        return child

    def probabilistic_mutation(self, individual):
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if np.random.rand() < self.hypernetwork[0]['mutation_probability']:
                mutated_individual[i] += np.random.uniform(-1.0, 1.0)
                mutated_individual[i] = max(-5.0, min(5.0, mutated_individual[i]))
        return mutated_individual

    def elitism(self, mutated_offspring):
        best_individuals = [min(mutated_offspring, key=operator.itemgetter(0))]
        best_individuals.extend(mutated_offspring[:int(self.population_size * 0.1)])
        return best_individuals

    def optimize(self, func):
        for _ in range(self.budget):
            evaluations = self.evaluate(func)
            selected_individuals = self.selection(evaluations)
            offspring = [self.probabilistic_crossover(selected_individuals[i], selected_individuals[j]) for i in range(len(selected_individuals)) for j in range(i+1, len(selected_individuals))]
            mutated_offspring = [self.probabilistic_mutation(offspring[i]) for i in range(len(offspring))]
            best_individuals = self.elitism(mutated_offspring)
            self.population = best_individuals

        best_individual = min(self.population, key=operator.itemgetter(0))
        return best_individual

# Example usage:
def func(x):
    return np.sum(x**2)

cpnhn = CPHHN(budget=100, dim=5)
best_individual = cpnhn.optimize(func)
print(best_individual)