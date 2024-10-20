import random
import numpy as np

class EvolutionaryBlackBoxOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            dim = self.dim * random.random()
            func = self.generate_func(dim)
            population.append((func, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)))
        return population

    def generate_func(self, dim):
        return np.sin(np.sqrt(dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = self.evaluate(func)
            if fitness < 0:
                break
        return func, fitness

    def evaluate(self, func):
        return func(func, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))

# One-line description: Evolutionary Black Box Optimization using Genetic Algorithm

class GeneticBlackBoxOptimization(EvolutionaryBlackBoxOptimization):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def mutate(individual):
            if random.random() < 0.01:
                dim = self.dim * random.random()
                func = self.generate_func(dim)
                individual = (func, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
            return individual
        def crossover(parent1, parent2):
            if random.random() < 0.5:
                dim = self.dim * random.random()
                func1 = self.generate_func(dim)
                func2 = self.generate_func(dim)
                individual1 = (func1, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
                individual2 = (func2, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
                individual = (individual1, individual2)
            else:
                individual = (parent1, parent2)
            return individual
        def selection(population):
            fitnesses = [self.evaluate(individual) for individual in population]
            return [individual for _, fitness in sorted(zip(population, fitnesses), reverse=True)[:self.population_size]]
        population = self.initialize_population()
        population = selection(population)
        while len(population) < self.population_size:
            parent1, parent2 = random.sample(population, 2)
            parent = crossover(parent1, parent2)
            population.append(mutate(parent))
        return population

# One-line description: Evolutionary Black Box Optimization using Genetic Algorithm

An exception occured: Traceback (most recent call last):
  File "/gpfs/scratch1/shared/hyin/LLaMEA/llamea/llamea.py", line 180, in initialize_single
    new_individual = self.evaluate_fitness(new_individual)
  File "/gpfs/scratch1/shared/hyin/LLaMEA/mutation_exp.py", line 44, in evaluateBBOB
  File "<string>", line 24, in __call__
  File "<string>", line 30, in evaluate
TypeError: __call__(): incompatible function arguments. The following argument types are supported:
    1. (self: ioh.iohcpp.problem.RealSingleObjective, arg0: List[float]) -> float
    2. (self: ioh.iohcpp.problem.RealSingleObjective, arg0: List[List[float]]) -> List[float]

Invoked with: <RealSingleObjectiveProblem 1. Sphere (iid=1 dim=5)>, <RealSingleObjectiveProblem 1. Sphere (iid=1 dim=5)>, -0.7096598752675742, -3.069158013315685