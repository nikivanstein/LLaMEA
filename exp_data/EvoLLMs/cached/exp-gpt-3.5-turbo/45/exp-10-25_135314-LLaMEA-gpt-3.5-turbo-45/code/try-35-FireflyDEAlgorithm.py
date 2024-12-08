import numpy as np

class FireflyDEAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=1.0, F=0.5, CR=0.9):
        super().__init__(budget, dim, alpha, beta0, gamma)
        self.F = F
        self.CR = CR

    def mutate(self, population, best_idx):
        idxs = list(range(self.budget))
        idxs.remove(best_idx)
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        return np.clip(mutant, -5.0, 5.0)

    def crossover(self, target, mutant):
        trial = np.copy(target)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() > self.CR and j != j_rand:
                trial[j] = mutant[j]
        return trial

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        light_intensities = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            for i in range(self.budget):
                mutant = self.mutate(population, i)
                trial = self.crossover(population[i], mutant)
                f_trial = func(trial)
                if f_trial < light_intensities[i]:
                    population[i] = trial
                    light_intensities[i] = f_trial

        return population[np.argmin(light_intensities)]