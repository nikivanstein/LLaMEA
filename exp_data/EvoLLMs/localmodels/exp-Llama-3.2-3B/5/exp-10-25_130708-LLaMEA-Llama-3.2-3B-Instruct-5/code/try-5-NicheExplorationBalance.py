import numpy as np
from scipy.optimize import minimize

class NicheExplorationBalance:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.niche_size = 5
        self.mutation_prob = 0.05
        self.niche_count = 0
        self.niche_centers = []
        self.fitness_history = []

    def __call__(self, func):
        # Initialize population with random niches
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        for i in range(self.population_size):
            self.niche_count += 1
            if self.niche_count > self.niche_size:
                self.niche_centers.append(self.niche_size)
                self.niche_count = 1
            population[i] = self.niche_centers[np.random.randint(0, self.niche_count)]

        # Evaluate population and store fitness history
        fitness_history = []
        for x in population:
            f = func(x)
            fitness_history.append(f)
            self.fitness_history.append(f)

        # Main loop
        for _ in range(self.budget):
            # Select fittest individuals
            fittest_individuals = np.argsort(fitness_history)[-self.population_size:]
            fittest_population = population[fittest_individuals]

            # Calculate niches
            niches = np.array_split(fittest_population, self.niche_count)
            niches = np.array([np.mean(niche, axis=0) for niche in niches])

            # Update niche centers with probability
            if np.random.rand() < self.mutation_prob:
                self.niche_centers = niches[np.random.randint(0, len(self.niche_centers))]
                self.niche_centers = np.random.uniform(-5.0, 5.0, size=(self.niche_count, self.dim))

            # Evaluate new population
            new_population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_prob:
                    new_population[i] = self.niche_centers[np.random.randint(0, self.niche_count)]
                else:
                    new_population[i] = self.niche_centers[np.random.randint(0, len(self.niche_centers))]

            # Store new fitness history
            new_fitness_history = []
            for x in new_population:
                f = func(x)
                new_fitness_history.append(f)
                self.fitness_history.append(f)

            # Update population and fitness history
            population = new_population
            fitness_history = new_fitness_history

# Example usage
def func(x):
    return np.sum(x**2)

def evaluateBBOB(func, population):
    # Evaluate population on BBOB test suite
    results = []
    for i in range(24):
        population = np.array([np.random.uniform(-5.0, 5.0, size=(100, 10))])
        population = evaluate_func(func, population)
        results.append(func(population[0]))
    return np.mean(results)

def evaluate_func(func, population):
    # Evaluate population on BBOB test suite
    population = np.array(population)
    results = []
    for i in range(24):
        results.append(func(population[i]))
    return population[np.argmin(results)]

det = NicheExplorationBalance(budget=100, dim=10)
det(func)
print(evaluateBBOB(det.func, det.population_size * np.random.uniform(-5.0, 5.0, size=(100, 10))))