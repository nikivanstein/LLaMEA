import numpy as np

class EnhancedFireflyLevyAlgorithmRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.2
        self.beta_min = 0.2
        self.gamma = 1.5

    def levy_flight(self):
        sigma = (np.math.gamma(1 + self.gamma) * np.math.sin(np.pi * self.gamma / 2) / (np.math.gamma((1 + self.gamma) / 2) * self.gamma * 2 ** ((self.gamma - 1) / 2))) ** (1 / self.gamma)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / self.gamma)
        return step

    def particle_swarm_mutation(self, population, i, j):
        delta = np.random.uniform(0, 1, self.dim)
        return delta * (population[j] - population[i])

    def dynamic_mutation_scale(self, fitness):
        return 0.1 + 0.4 * (1 - np.tanh(np.mean(fitness)))

    def __call__(self, func):
        population_size = 50
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = [func(individual) for individual in population]

        for _ in range(self.budget):
            for i in range(population_size):
                for j in range(population_size):
                    if fitness[i] > fitness[j]:
                        r = np.linalg.norm(population[i] - population[j])
                        beta = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * r ** 2)
                        step = self.levy_flight()
                        mutation = self.particle_swarm_mutation(population, i, j)
                        population[i] += beta * (population[j] - population[i]) + self.alpha * step + mutation * self.dynamic_mutation_scale(fitness)
                        population[i] = np.clip(population[i], -5.0, 5.0)
                        new_fitness = func(population[i])
                        if new_fitness < fitness[i]:
                            fitness[i] = new_fitness

            alpha_min = 0.1
            alpha_max = 0.5
            self.alpha = alpha_min + (alpha_max - alpha_min) * (_ / self.budget)

            dynamic_population_size = int(40 + 10 * np.sin(_ / self.budget * np.pi))
            if dynamic_population_size != population_size:
                if dynamic_population_size > population_size:
                    new_population = np.random.uniform(-5.0, 5.0, (dynamic_population_size - population_size, self.dim))
                    population = np.vstack([population, new_population])
                    fitness.extend([func(individual) for individual in new_population])
                else:
                    indices_to_remove = np.random.choice(range(len(population)), population_size - dynamic_population_size, replace=False)
                    population = np.delete(population, indices_to_remove, axis=0)
                    fitness = [fitness[i] for i in range(len(fitness)) if i not in indices_to_remove]
                population_size = dynamic_population_size

        best_index = np.argmin(fitness)
        return population[best_index]