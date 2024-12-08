import numpy as np

class HybridFireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=1.0, de_weight=0.5, de_cr=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.de_weight = de_weight
        self.de_cr = de_cr

    def attractiveness(self, light_intensity, distance):
        return self.beta0 * np.exp(-self.gamma * distance**2)

    def move_firefly(self, current, best, attractiveness):
        step = self.alpha * (np.random.rand(self.dim) - 0.5)
        return current + attractiveness * (best - current) + step

    def de_mutate(self, population, i):
        candidates = [idx for idx in range(self.budget) if idx != i]
        a, b, c = population[np.random.choice(candidates, 3, replace=False)]
        return population[i] + self.de_weight * (a - population[i]) + self.de_weight * (b - c)

    def de_crossover(self, target, mutant):
        crossover_points = np.random.rand(self.dim) < self.de_cr
        trial = np.where(crossover_points, mutant, target)
        return trial

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        light_intensities = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if light_intensities[j] > light_intensities[i]:
                        distance = np.linalg.norm(population[i] - population[j])
                        attractiveness_ij = self.attractiveness(light_intensities[j], distance)
                        population[i] = self.move_firefly(population[i], population[j], attractiveness_ij)
                        light_intensities[i] = func(population[i])
                        
                        mutant = self.de_mutate(population, i)
                        trial = self.de_crossover(population[i], mutant)
                        f_val = func(trial)
                        
                        if f_val < light_intensities[i]:
                            population[i] = trial
                            light_intensities[i] = f_val

        return population[np.argmin(light_intensities)]