import numpy as np

class HybridDELevyDynamic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.8
        self.CR = 0.9
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.func_evals = 0
        self.dynamic_factor = 0.5  # Factor for dynamic population changes

    def levy_flight(self, L):
        u = np.random.normal(0, 1) * (L ** (-1/3))
        v = np.random.normal(0, 1)
        step = u / (abs(v) ** (1/3))
        return step

    def dynamic_population_size(self):
        new_size = int(self.population_size * (1 - self.dynamic_factor * self.func_evals / self.budget))
        return max(5, new_size)

    def __call__(self, func):
        best = None
        best_score = np.inf
        
        def evaluate(ind):
            nonlocal best, best_score
            if self.func_evals < self.budget:
                score = func(ind)
                self.func_evals += 1
                if score < best_score:
                    best_score = score
                    best = ind
                return score
            else:
                return None

        for i in range(self.population_size):
            self.scores[i] = evaluate(self.population[i])

        while self.func_evals < self.budget:
            new_population_size = self.dynamic_population_size()
            if new_population_size < self.population_size:
                indices = np.argsort(self.scores)[:new_population_size]
                self.population = self.population[indices]
                self.scores = self.scores[indices]
            self.population_size = new_population_size

            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])
                levy_step = self.levy_flight(1.5) * (trial - best)
                trial = trial + levy_step
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                trial_score = evaluate(trial)
                if trial_score is not None and trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

        return best, best_score