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
                'elitism_rate': np.random.uniform(0.0, 1.0),
                'crossover_probability_adaptive': False,
               'mutation_probability_adaptive': False
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

    def crossover(self, selected_individuals):
        offspring = []
        for _ in range(self.population_size - len(selected_individuals)):
            parent1, parent2 = random.sample(selected_individuals, 2)
            crossover_point = np.random.randint(1, self.dim)
            if self.hypernetwork[0]['crossover_probability_adaptive']:
                if np.random.rand() < self.hypernetwork[0]['crossover_probability']:
                    child = (parent1[:crossover_point] + parent2[crossover_point:]) if np.random.rand() < self.hypernetwork[0]['crossover_probability'] else parent1
                else:
                    child = parent1
            else:
                child = (parent1[:crossover_point] + parent2[crossover_point:]) if np.random.rand() < self.hypernetwork[0]['crossover_probability'] else parent1
            offspring.append(child)
        return offspring

    def mutation(self, offspring):
        mutated_offspring = []
        for individual in offspring:
            for i in range(self.dim):
                if np.random.rand() < self.hypernetwork[0]['mutation_probability_adaptive']:
                    individual[i] += np.random.uniform(-1.0, 1.0)
                    individual[i] = max(-5.0, min(5.0, individual[i]))
                else:
                    individual[i] += np.random.uniform(-1.0, 1.0)
                    individual[i] = max(-5.0, min(5.0, individual[i]))
            mutated_offspring.append(individual)
        return mutated_offspring

    def elitism(self, mutated_offspring):
        best_individuals = [min(mutated_offspring, key=operator.itemgetter(0))]
        best_individuals.extend(mutated_offspring[:int(self.population_size * 0.1)])
        return best_individuals

    def optimize(self, func):
        for _ in range(self.budget):
            evaluations = self.evaluate(func)
            selected_individuals = self.selection(evaluations)
            offspring = self.crossover(selected_individuals)
            mutated_offspring = self.mutation(offspring)
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