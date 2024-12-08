import numpy as np
from scipy.optimize import minimize

class NicheEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.niche_size = 5
        self.population_size = 100
        self.niche_count = 0
        self.niche_centers = []
        self.fitness_history = []
        self.perturbation_rate = 0.05

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

            # Update niche centers
            self.niche_centers = niches

            # Perturb niche centers
            perturbed_niche_centers = self.perturb_niche_centers(self.niche_centers)

            # Evaluate new population
            new_population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            for i in range(self.population_size):
                new_population[i] = perturbed_niche_centers[np.random.randint(0, self.niche_count)]

            # Store new fitness history
            new_fitness_history = []
            for x in new_population:
                f = func(x)
                new_fitness_history.append(f)
                self.fitness_history.append(f)

            # Update population and fitness history
            population = new_population
            fitness_history = new_fitness_history

    def perturb_niche_centers(self, niche_centers):
        perturbed_niche_centers = niche_centers.copy()
        for i in range(len(perturbed_niche_centers)):
            perturbation = np.random.uniform(-1, 1, self.dim)
            perturbed_niche_centers[i] += perturbation * self.perturbation_rate
            perturbed_niche_centers[i] = np.clip(perturbed_niche_centers[i], -5.0, 5.0)
        return perturbed_niche_centers

# Example usage
def func(x):
    return np.sum(x**2)

ne = NicheEvolution(budget=100, dim=10)
ne(func)