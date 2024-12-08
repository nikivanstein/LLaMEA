import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim, alpha=0.1, beta=0.5):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.alpha = alpha
        self.beta = beta
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            self.func_evaluations += 1
            new_individual = np.copy(self.population)
            for i in range(self.population_size):
                for j in range(self.dim):
                    new_individual[j] += np.random.normal(0, 1) * (func(new_individual[j]) - func(new_individual[i]))
                    new_individual[j] = np.clip(new_individual[j], -5.0, 5.0)
                new_individual[i] = np.clip(new_individual[i], -5.0, 5.0)
            new_individual = np.sort(new_individual, axis=0)
            self.population = new_individual
            self.fitnesses = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                self.fitnesses[i] = func(self.population[i])

        return self.fitnesses

    def select(self, individuals, probabilities):
        return np.random.choice(len(individuals), size=self.population_size, p=probabilities)

    def mutate(self, individuals, probabilities):
        return self.select(individuals, probabilities)

    def crossover(self, parents, probabilities):
        offspring = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            parent1, parent2 = np.random.choice(len(parents), size=2, p=probabilities)
            offspring[i] = np.clip(parents[parent1] + parents[parent2] * (self.alpha * (parents[parent1] - parents[parent2]) ** self.beta, -5.0, 5.0)
        return offspring

    def evaluateBBOB(self, func):
        for i in range(self.population_size):
            self.fitnesses[i] = func(self.population[i])