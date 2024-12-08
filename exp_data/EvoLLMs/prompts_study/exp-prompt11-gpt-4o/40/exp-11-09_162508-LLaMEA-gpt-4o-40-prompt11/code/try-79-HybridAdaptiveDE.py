import numpy as np

class HybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 40  # Increased initial population for greater diversity
        self.population_size = int(self.initial_population_size * 1.25)  # Slightly larger population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.6  # Adjusted mutation factor for improved convergence
        self.CR = 0.8  # Balanced crossover probability for exploration and exploitation
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.func_evals = 0

    def levy_flight(self):
        beta = 1.5
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, 1)
        step = u / (abs(v) ** (1 / beta))
        return step

    def adapt_parameters(self):
        self.F = 0.5 + 0.5 * np.random.rand()  # Dynamic mutation factor for varied search behavior
        self.CR = 0.6 + 0.4 * np.random.rand()  # Flexible crossover range for adaptability

    def enhanced_mutation(self, best, idx):
        if np.random.rand() < 0.6:
            # DE/rand/2 mutation
            indices = list(range(self.population_size))
            indices.remove(idx)
            a, b, c, d, e = np.random.choice(indices, 5, replace=False)
            return self.population[a] + self.F * (self.population[b] - self.population[c] + self.population[d] - self.population[e])
        else:
            # DE/best/1 mutation
            indices = list(range(self.population_size))
            indices.remove(idx)
            a, b = np.random.choice(indices, 2, replace=False)
            return best + self.F * (self.population[a] - self.population[b])

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

                # Possible Lévy flight step
                levy_step = self.levy_flight() * (trial - best)
                trial = trial + levy_step
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Selection
                trial_score = evaluate(trial)
                if trial_score is not None and trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

        return best, best_score