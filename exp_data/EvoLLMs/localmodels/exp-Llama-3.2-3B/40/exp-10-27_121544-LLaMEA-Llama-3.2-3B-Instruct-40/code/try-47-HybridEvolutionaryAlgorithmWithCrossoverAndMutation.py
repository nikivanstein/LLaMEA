import numpy as np
import random

class HybridEvolutionaryAlgorithmWithCrossoverAndMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        # Initialize population
        population = [np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim) for _ in range(100)]

        for _ in range(self.budget):
            # Evaluate fitness
            fitness = [func(individual) for individual in population]

            # Select parents using tournament selection
            parents = []
            for _ in range(10):
                tournament = random.sample(population, 3)
                parents.append(min(tournament, key=lambda x: fitness[tournament.index(x)]))

            # Perform crossover and mutation
            new_population = []
            for _ in range(10):
                parent1, parent2 = random.sample(parents, 2)
                child = []
                for i in range(self.dim):
                    if random.random() < 0.4:
                        child.append((parent1[i] + parent2[i]) / 2)
                    else:
                        child.append(random.choice([parent1[i], parent2[i]]))
                new_population.append(child)

            # Replace least fit individual
            new_population = sorted(new_population, key=lambda x: func(x))[:10]
            population = new_population

        # Return best individual
        return min(population, key=lambda x: func(x))

# Example usage:
def f1(x):
    return sum(x**2)

def f2(x):
    return sum(x**2) + sum(y**2 for y in x)

def f3(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x)

def f4(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x)

def f5(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x)

def f6(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x)

def f7(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x)

def f8(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x)

def f9(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x)

def f10(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x)

def f11(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x)

def f12(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x)

def f13(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x)

def f14(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x)

def f15(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x)

def f16(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x)

def f17(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x)

def f18(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x)

def f19(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x)

def f20(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x)

def f21(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x)

def f22(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x)

def f23(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x)

def f24(x):
    return sum(x**2) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(w**2 for w in x) + sum(u**2 for u in x) + sum(v**2 for v in x) + sum(w**2 for w in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x) + sum(z**2 for z in x) + sum(x**2 for x in x) + sum(y**2 for y in x)

hybrid_evolutionary_algorithm_with_crossover_and_mutation = HybridEvolutionaryAlgorithmWithCrossoverAndMutation(budget=100, dim=10)

def optimize_bbob(func):
    return hybrid_evolutionary_algorithm_with_crossover_and_mutation(func)

# Example usage:
def f1(x):
    return sum(x**2)

result = optimize_bbob(f1)
print(result)