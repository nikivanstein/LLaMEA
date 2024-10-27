import numpy as np
import random
import copy

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        def refine_strategy(individual):
            strategy = copy.deepcopy(individual['strategy'])
            for i in range(self.dim):
                if random.random() < 0.4:
                    strategy[i] = random.choice(['sbx', 'rand1', 'uniform'])
            return {'individual': individual,'strategy': strategy}

        def evaluateBBOB(func, individual):
            code = f"def f(x):\n    return {func.__name__}(x)"
            exec(code, globals())
            result = eval('f(individual["x"])')
            individual['fitness'] = -result
            return individual

        def crossover(parent1, parent2):
            child = {}
            for i in range(self.dim):
                if random.random() < 0.5:
                    child['x'][i] = parent1['x'][i]
                else:
                    child['x'][i] = parent2['x'][i]
            child['strategy'] = parent1['strategy']
            child['fitness'] = parent1['fitness']
            return child

        def mutate(individual):
            if random.random() < 0.1:
                individual['x'][random.randint(0, self.dim-1)] += np.random.uniform(-1, 1)
                individual['x'][random.randint(0, self.dim-1)] = max(self.bounds[0][0], min(individual['x'][random.randint(0, self.dim-1)], self.bounds[0][1]))
            return individual

        population = [{'x': self.x0,'strategy':'sbx', 'fitness': 0} for _ in range(10)]
        for _ in range(self.budget):
            population = sorted(population, key=lambda x: x['fitness'])
            if population[0]['fitness']!= 0:
                individual = population[0]
                new_individual = evaluateBBOB(func, individual)
                population[0] = new_individual
                if random.random() < 0.5:
                    new_individual = crossover(population[1], population[0])
                    population[1] = new_individual
                if random.random() < 0.1:
                    new_individual = mutate(population[1])
                    population[1] = new_individual
            else:
                break
        return population[0]['x'], population[0]['fitness']

# Example usage:
budget = 100
dim = 10
func = lambda x: x[0]**2 + x[1]**2
algorithm = HybridEvolutionaryAlgorithm(budget, dim)
result = algorithm(func)
print(result)