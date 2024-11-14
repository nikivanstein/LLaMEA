import numpy as np

class HybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 40  # Further increased initial population for more diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.scores = np.full(self.initial_population_size, np.inf)
        self.func_evals = 0
        self.dynamic_population_size = self.initial_population_size
        self.F = 0.5
        self.CR = 0.9

    def levy_flight(self, L):
        beta = 1.5  # Adjusted for exploration-exploitation balance
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, 1)
        step = u / (abs(v) ** (1 / beta))
        return step * np.random.normal(0, 1, self.dim)

    def adapt_parameters(self):
        self.F = 0.3 + 0.5 * np.random.rand()  # Further variation for adaptive mutation
        self.CR = 0.6 + 0.4 * np.random.rand()  # Increased range for crossover

    def multi_strategy_mutation(self, best_idx):
        indices = list(range(self.dynamic_population_size))
        a, b, c = np.random.choice(indices, 3, replace=False)
        if np.random.rand() < 0.6:  # Favor DE/rand/1
            return self.population[a] + self.F * (self.population[b] - self.population[c])
        else:
            return self.population[best_idx] + self.F * (self.population[a] - self.population[b])  # DE/best/1

    def dynamic_population_resize(self):
        # Gradually reduce population size for focused search
        self.dynamic_population_size = max(4, int(self.initial_population_size * (1 - 0.5 * (self.func_evals / self.budget))))

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
            self.dynamic_population_resize()
            for i in range(self.dynamic_population_size):
                best_idx = np.argmin(self.scores[:self.dynamic_population_size])
                mutant = self.multi_strategy_mutation(best_idx)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                levy_step = self.levy_flight(1.5)
                trial = trial + levy_step
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                trial_score = evaluate(trial)
                if trial_score is not None and trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

        return best, best_score