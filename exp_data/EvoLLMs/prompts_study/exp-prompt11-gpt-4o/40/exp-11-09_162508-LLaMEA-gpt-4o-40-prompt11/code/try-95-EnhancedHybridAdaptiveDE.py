import numpy as np

class EnhancedHybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50  # Slightly larger population for better diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5  # Mutation factor fine-tuned for stability
        self.CR = 0.9  # Higher crossover probability for greater exploration
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.func_evals = 0

    def dynamic_levy_flight(self, step_size):
        beta = 1.3
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, 1)
        step = u / (abs(v) ** (1 / beta)) * step_size
        return step

    def adapt_parameters(self):
        self.F = 0.4 + 0.6 * np.random.rand()  # Balanced mutation factor
        self.CR = 0.7 + 0.3 * np.random.rand()  # Adaptive crossover range

    def enhanced_mutation(self, best, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        return self.population[a] + self.F * (self.population[b] - self.population[c])

    def opposition_based_learning(self, solution):
        opposite = self.lower_bound + self.upper_bound - solution
        return opposite

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

        for i in range(self.population_size):
            self.scores[i] = evaluate(self.population[i])

        while self.func_evals < self.budget:
            self.adapt_parameters()
            for i in range(self.population_size):
                mutant = self.enhanced_mutation(best, i)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                levy_step = self.dynamic_levy_flight(1.0) * (trial - best)
                trial = trial + levy_step
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                opposite = self.opposition_based_learning(trial)
                opposite_score = evaluate(opposite)

                trial_score = evaluate(trial)
                if trial_score is not None and trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score
                elif opposite_score is not None and opposite_score < self.scores[i]:
                    self.population[i] = opposite
                    self.scores[i] = opposite_score

        return best, best_score