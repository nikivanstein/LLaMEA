import numpy as np

class EnhancedHybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50  # Slightly larger population for better exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5  # Adjusted mutation factor for balance
        self.CR = 0.9  # Increased crossover probability for exploration
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.func_evals = 0

    def levy_flight(self):
        beta = 1.5
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return u / (np.abs(v) ** (1 / beta))

    def adapt_parameters(self):
        self.F = 0.4 + 0.6 * np.random.rand()  # Wider range for mutation factor
        self.CR = 0.7 + 0.3 * np.random.rand()  # Narrower range for crossover

    def enhanced_mutation(self, best, idx):
        indices = np.arange(self.population_size)
        indices = indices[indices != idx]
        if np.random.rand() < 0.3:
            # DE/best/2 mutation
            a, b, c, d = np.random.choice(indices, 4, replace=False)
            return best + self.F * (self.population[a] - self.population[b] + self.population[c] - self.population[d])
        else:
            # DE/rand/1 mutation
            a, b, c = np.random.choice(indices, 3, replace=False)
            return self.population[a] + self.F * (self.population[b] - self.population[c])

    def dynamic_local_search(self, candidate, best):
        scale = np.random.uniform(0.1, 0.3, self.dim)  # Small local search step
        return candidate + scale * (best - candidate)

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

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                if np.random.rand() < 0.2:
                    # Apply dynamic local search with a certain probability
                    trial = self.dynamic_local_search(trial, best)

                # Lévy flight step
                levy_step = self.levy_flight() * (trial - best)
                trial = trial + levy_step
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Selection
                trial_score = evaluate(trial)
                if trial_score is not None and trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

        return best, best_score