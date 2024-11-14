import numpy as np

class AdaptiveDELS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.lb = -5.0
        self.ub = 5.0
        self.F = 0.8  # Scaling factor for mutation
        self.CR = 0.7  # Crossover probability

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])

        best_idx = np.argmin(scores)
        best_solution = population[best_idx]
        best_score = scores[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                x1, x2, x3 = population[idxs]
                mutant = np.clip(x1 + self.F * (x2 - x3), self.lb, self.ub)

                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                trial_score = func(trial)
                evaluations += 1

                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

                    if trial_score < best_score:
                        best_solution = trial
                        best_score = trial_score

            if evaluations + self.population_size <= self.budget:
                for i in range(self.population_size):
                    local_search = best_solution + np.random.normal(0, 0.05, self.dim)
                    local_search = np.clip(local_search, self.lb, self.ub)
                    local_score = func(local_search)
                    evaluations += 1

                    if local_score < scores[i]:
                        population[i] = local_search
                        scores[i] = local_score

                        if local_score < best_score:
                            best_solution = local_search
                            best_score = local_score

        return best_solution, best_score