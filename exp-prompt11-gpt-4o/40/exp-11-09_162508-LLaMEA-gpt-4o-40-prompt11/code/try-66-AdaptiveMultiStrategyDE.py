import numpy as np

class AdaptiveMultiStrategyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30  # Increased initial population for diversity
        self.population_size = int(self.initial_population_size * 1.2)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5  # Adjusted mutation factor for more stability
        self.CR = 0.9  # Higher crossover probability to encourage exploration
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.func_evals = 0

    def levy_flight(self, L):
        beta = 1.7
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, 1)
        step = u / (abs(v) ** (1 / beta))
        return step

    def adapt_parameters(self):
        self.F = 0.4 + 0.4 * np.random.rand()  # Wider variation in mutation factor
        self.CR = 0.7 + 0.3 * np.random.rand()  # Adjusted crossover range

    def multi_strategy_mutation(self, best, idx):
        if np.random.rand() < 0.5:
            # DE/rand/1 mutation
            indices = list(range(self.population_size))
            indices.remove(idx)
            a, b, c = np.random.choice(indices, 3, replace=False)
            return self.population[a] + self.F * (self.population[b] - self.population[c])
        else:
            # DE/best/2 mutation
            indices = list(range(self.population_size))
            indices.remove(idx)
            best_indices = np.argsort(self.scores)[:5]  # Select top 5 as potential best
            a, b = np.random.choice(best_indices, 2, replace=False)
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

        for i in range(self.population_size):
            self.scores[i] = evaluate(self.population[i])

        while self.func_evals < self.budget:
            self.adapt_parameters()
            for i in range(self.population_size):
                mutant = self.multi_strategy_mutation(best, i)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Possible Lévy flight step
                levy_step = self.levy_flight(1.5) * (trial - best)
                trial = trial + levy_step
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Selection
                trial_score = evaluate(trial)
                if trial_score is not None and trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

        return best, best_score