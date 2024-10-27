import numpy as np

class DEHSAlgorithm:
    def __init__(self, budget, dim, population_size=50, cr=0.8, f=0.5, min_bandwidth=0.01, max_bandwidth=0.1, min_exp_prob=0.7, max_exp_prob=0.95):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.cr = cr
        self.f = f
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth
        self.min_exp_prob = min_exp_prob
        self.max_exp_prob = max_exp_prob

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def mutate_differential_evolution(self, population, target_idx):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        mutant = population[idxs[0]] + self.f * (population[idxs[1]] - population[idxs[2]])
        return mutant

    def improvise_harmony(self, harmony_memory, iteration):
        bandwidth = self.min_bandwidth + (self.max_bandwidth - self.min_bandwidth) * (iteration / self.budget)
        exp_prob = self.max_exp_prob - (self.max_exp_prob - self.min_exp_prob) * (iteration / self.budget)

        new_harmony = np.copy(harmony_memory[np.random.randint(self.population_size)])
        for i in range(self.dim):
            if np.random.rand() < bandwidth:
                new_harmony[i] = np.random.uniform(-5.0, 5.0)
            if np.random.rand() < exp_prob:
                diff = self.mutate_differential_evolution(harmony_memory, i)
                new_harmony[i] = harmony_memory[np.random.randint(self.population_size)][i] + np.random.uniform(0, 1) * diff[i]
        return new_harmony

    def __call__(self, func):
        population = self.initialize_population()
        population_fitness = np.array([func(individual) for individual in population])

        for iteration in range(self.budget):
            new_harmony = self.improvise_harmony(population, iteration)
            new_fitness = func(new_harmony)
            best_idx = np.argmin(population_fitness)
            
            if new_fitness < population_fitness[best_idx]:
                population[best_idx] = new_harmony
                population_fitness[best_idx] = new_fitness
        
        return population[np.argmin(population_fitness)]