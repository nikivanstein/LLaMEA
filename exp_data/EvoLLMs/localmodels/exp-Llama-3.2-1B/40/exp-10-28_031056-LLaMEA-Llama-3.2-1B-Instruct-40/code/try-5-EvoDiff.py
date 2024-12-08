import numpy as np

class EvoDiff:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.init_population()

    def init_population(self):
        # Initialize the population with random solutions
        return np.random.uniform(-5.0, 5.0, self.dim) + np.random.normal(0, 1, self.dim)

    def __call__(self, func):
        # Evaluate the black box function with the current population
        func_values = np.array([func(x) for x in self.population])

        # Select the fittest solutions
        fittest_indices = np.argsort(func_values)[::-1][:self.population_size]

        # Evolve the population using evolutionary differential evolution
        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = np.array([self.population[i] for i in fittest_indices])

            # Perform mutation
            mutated_parents = parents.copy()
            for _ in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    mutated_parents[_] += np.random.normal(0, 1, self.dim)

            # Select offspring using tournament selection
            offspring = np.array([self.population[i] for i in np.argsort(mutated_parents)[::-1][:self.population_size]])

            # Replace the old population with the new one
            self.population = np.concatenate((self.population, mutated_parents), axis=0)
            self.population = np.concatenate((self.population, offspring), axis=0)

        # Evaluate the function with the final population
        func_values = np.array([func(x) for x in self.population])
        return func_values

    def refine_strategy(self, func):
        # Refine the strategy by changing the individual lines of the selected solution
        # to refine its strategy
        for i, individual in enumerate(self.population):
            # Calculate the fitness of the individual
            fitness = func(individual)

            # If the fitness is low, change the individual line to improve its strategy
            if fitness < 0.4:
                # Select two parents using tournament selection
                parents = np.array([individual for j in range(2) if np.random.rand() < 0.5])

                # Perform mutation
                mutated_parents = parents.copy()
                for _ in range(self.population_size):
                    if np.random.rand() < self.mutation_rate:
                        mutated_parents[_] += np.random.normal(0, 1, self.dim)

                # Select offspring using tournament selection
                offspring = np.array([individual for j in range(2) if np.random.rand() < 0.5])

                # Replace the old population with the new one
                self.population = np.concatenate((self.population, mutated_parents), axis=0)
                self.population = np.concatenate((self.population, offspring), axis=0)

            # If the fitness is high, keep the individual line as it is
            else:
                pass

# Exception occured: Traceback (most recent call last):
#   File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
#     new_individual = self.evaluate_fitness(new_individual)
#   File "/root/LLaMEA/mutation_exp.py", line 52, in evaluateBBOB
#     algorithm(problem)
#   File "<string>", line 17, in __call__
#     File "<string>", line 17, in <listcomp>
#     TypeError: __call__(): incompatible function arguments. The following argument types are supported:
#         1. (self: ioh.iohcpp.problem.RealSingleObjective, arg0: List[float]) -> float
#         2. (self: ioh.iohcpp.problem.RealSingleObjective, arg0: List[List[float]]) -> List[float]
# Invoked with: <RealSingleObjectiveProblem 1. Sphere (iid=1 dim=5)>, -0.35458901455603975