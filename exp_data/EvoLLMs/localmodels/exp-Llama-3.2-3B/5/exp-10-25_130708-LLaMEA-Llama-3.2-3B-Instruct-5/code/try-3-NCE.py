import numpy as np

class NCE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.niche_size = 5
        self.population_size = 100
        self.niche_count = 0
        self.niche_centers = []
        self.fitness_history = []
        self.cluster_prob = 0.05

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

            # Cluster individuals
            cluster_individuals = []
            for i in range(self.population_size):
                if np.random.rand() < self.cluster_prob:
                    cluster_individuals.append(fittest_population[i])
                else:
                    cluster_individuals.append(np.random.uniform(-5.0, 5.0, size=self.dim))

            # Update population
            population = np.array(cluster_individuals)

            # Evaluate new population
            new_fitness_history = []
            for x in population:
                f = func(x)
                new_fitness_history.append(f)
                self.fitness_history.append(f)

# Example usage
def func(x):
    return np.sum(x**2)

nce = NCE(budget=100, dim=10)
nce(func)