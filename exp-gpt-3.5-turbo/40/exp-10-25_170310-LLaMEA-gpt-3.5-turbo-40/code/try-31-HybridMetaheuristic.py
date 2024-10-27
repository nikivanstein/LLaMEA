import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.pso_inertia_weight = 0.7
        self.pso_cognitive_weight = 1.5
        self.pso_social_weight = 2.0
        self.de_scale_factor = 0.5
        self.de_crossover_prob = 0.9

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                # PSO update
                pbest = self.population[np.argmin([func(p) for p in self.population])]
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    self.population[i][j] = (self.pso_inertia_weight * self.population[i][j] +
                                             self.pso_cognitive_weight * r1 * (pbest[j] - self.population[i][j]) +
                                             self.pso_social_weight * r2 * (self.population[i][j] - self.population[np.random.randint(self.budget)][j]))
                # DE update
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = self.population[a] + self.de_scale_factor * (self.population[b] - self.population[c])
                crossover_mask = np.random.rand(self.dim) < self.de_crossover_prob
                trial = np.where(crossover_mask, mutant, self.population[i])
                fitness_trial = func(trial)
                if fitness_trial < func(self.population[i]):
                    self.population[i] = trial
        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]

        return best_solution