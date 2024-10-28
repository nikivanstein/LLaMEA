import random
import numpy as np

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Initialize population with random individuals
        population = [self.__init__(self.budget, self.dim) for _ in range(100)]

        # Evolve population over iterations
        for _ in range(1000):
            # Select parents using tournament selection
            parents = []
            for _ in range(100):
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                tournament = [x for x in range(len(parent1.search_space)) if random.random() < 0.5]
                tournament_parent = [x for x in range(len(parent1.search_space)) if x not in tournament]
                tournament_parent.sort(key=lambda i: random.random(), reverse=True)
                tournament_parent = [tournament_parent[i] for i in range(len(tournament_parent))]
                tournament_parent[0] = parent1.search_space[tournament_parent[0]]
                tournament_parent[1] = parent2.search_space[tournament_parent[1]]
                parents.append(self.__call__(func, tournament_parent))

            # Crossover (recombination) parents
            offspring = []
            for _ in range(100):
                parent1, parent2 = random.sample(parents, 2)
                child = parent1.search_space[:len(parent1.search_space)//2] + parent2.search_space[len(parent2.search_space)//2:]
                offspring.append(self.__call__(func, child))

            # Mutate offspring
            for individual in offspring:
                if random.random() < 0.1:
                    index = random.randint(0, len(individual.search_space) - 1)
                    individual.search_space[index] = random.uniform(-5.0, 5.0)

        # Return best individual
        return max(set(population), key=lambda x: x.f(func))

# Initialize algorithm
algorithm = NovelMetaheuristicAlgorithm(100, 10)

# Run algorithm on BBOB test suite
results = []
for func in ["linear", "sphere", "cube", "quadratic", "exponential", "logistic", "triangular", "tanh"]:
    best_individual = algorithm(algorithm, func)
    results.append((best_individual, func, algorithm(algorithm, func)))

# Print results
print("Results:")
for name, description, score in results:
    print(f"{name}: {description} - Score: {score}")