import numpy as np

class EnhancedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 50
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.scores = np.full(self.initial_population_size, np.inf)
        self.func_evals = 0
        self.F = 0.7
        self.CR = 0.9

    def levy_flight(self, scale=0.01):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / beta))
        return scale * step

    def adapt_parameters(self):
        self.F = np.random.uniform(0.5, 0.9)
        self.CR = np.random.uniform(0.7, 0.95)

    def resized_population(self):
        proportion = min(1, max(0.5, (self.budget - self.func_evals) / self.budget))
        new_size = int(self.initial_population_size * proportion)
        if new_size < len(self.population):
            indices = np.argsort(self.scores)[:new_size]
            self.population = self.population[indices]
            self.scores = self.scores[indices]

    def enhanced_mutation(self, best, idx):
        indices = np.random.choice(np.delete(np.arange(len(self.population)), idx), 5, replace=False)
        if np.random.rand() < 0.7:
            return self.population[indices[0]] + self.F * (self.population[indices[1]] - self.population[indices[2]] + self.population[indices[3]] - self.population[indices[4]])
        else:
            return best + self.F * (self.population[indices[0]] - self.population[indices[1]])

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
                    best = ind.copy()
                return score
            else:
                return None

        for i in range(len(self.population)):
            self.scores[i] = evaluate(self.population[i])

        while self.func_evals < self.budget:
            self.adapt_parameters()
            self.resized_population()
            for i in range(len(self.population)):
                mutant = self.enhanced_mutation(best, i)

                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                levy_step = self.levy_flight() * (trial - best)
                trial = trial + levy_step
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                trial_score = evaluate(trial)
                if trial_score is not None and trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

        return best, best_score