import numpy as np

class HybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 40  # Increased population for diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.6  # Adjusted mutation factor for balanced exploration
        self.CR = 0.8  # Balanced crossover probability
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.scores = np.full(self.initial_population_size, np.inf)
        self.func_evals = 0

    def levy_flight(self, L):
        beta = 1.5
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, 1)
        step = u / (abs(v) ** (1 / beta))
        return step

    def adapt_parameters(self):
        self.F = 0.5 + 0.3 * np.random.rand()  # Slightly narrower mutation factor range
        self.CR = 0.6 + 0.4 * np.random.rand()  # Increased crossover range

    def competitive_selection(self, trial_score, current_score):
        return trial_score < current_score

    def multi_strategy_mutation(self, best, idx):
        if np.random.rand() < 0.4:
            # DE/rand/1 mutation
            indices = list(range(self.initial_population_size))
            indices.remove(idx)
            a, b, c = np.random.choice(indices, 3, replace=False)
            return self.population[a] + self.F * (self.population[b] - self.population[c])
        else:
            # DE/best/1 with competitive mutation
            indices = list(range(self.initial_population_size))
            indices.remove(idx)
            best_indices = np.argsort(self.scores)[:5]  # Select top 5 as potential best
            a = np.random.choice(best_indices)
            c, d = np.random.choice(indices, 2, replace=False)
            return best + self.F * (self.population[a] - self.population[c] + self.population[b] - self.population[d])

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

        for i in range(self.initial_population_size):
            self.scores[i] = evaluate(self.population[i])

        while self.func_evals < self.budget:
            self.adapt_parameters()
            for i in range(self.initial_population_size):
                mutant = self.multi_strategy_mutation(best, i)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Lévy flight step for global exploration
                levy_step = self.levy_flight(1.5) * (trial - best)
                trial = trial + levy_step
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Selection with competitive selection strategy
                trial_score = evaluate(trial)
                if trial_score is not None and self.competitive_selection(trial_score, self.scores[i]):
                    self.population[i] = trial
                    self.scores[i] = trial_score

        return best, best_score