import numpy as np
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.es_params = {
            'population_size': 10,
           'mutation_rate': 0.1,
            'crossover_rate': 0.5
        }

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Evolution Strategy (ES)
            x = self.es_params['population_size'] * np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))

            # Evaluate the ES
            f_es = func(x)

            # Update the best solution
            f_evals = f_es[0]
            x_best = x[0]
            f_evals_best = f_evals

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Evolutionary Algorithm (EA)
            population = [x for _ in range(self.es_params['population_size'])]
            for _ in range(10):
                # Selection
                parents = np.random.choice(population, size=self.es_params['population_size'], replace=False)

                # Crossover
                offspring = []
                for _ in range(self.es_params['population_size']):
                    parent1, parent2 = random.sample(parents, 2)
                    crossover_point = random.randint(1, self.dim - 1)
                    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                    offspring.append(child)

                # Mutation
                for i in range(self.es_params['population_size']):
                    if random.random() < self.es_params['mutation_rate']:
                        mutation = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))
                        offspring[i] += mutation

                # Replacement
                population = offspring

            # Evaluate the EA
            f_ea = func(population[0])

            # Update the best solution
            f_evals = f_ea[0]
            x_best = population[0]
            f_evals_best = f_evals

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Update the bounds
            self.bounds = np.array([np.min(population, axis=0), np.max(population, axis=0)])

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

he = HybridEvolutionaryAlgorithm(budget=10, dim=2)
x_opt = he(func)
print(x_opt)