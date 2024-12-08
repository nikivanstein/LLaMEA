import numpy as np

class HybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50  # Increased initial population for enhanced exploration
        self.population_size = int(self.initial_population_size * 1.2)  # Slightly adjusted population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.7  # Increased mutation factor for broader search
        self.CR = 0.7  # Adjusted crossover probability for balance
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.func_evals = 0

    def dual_levy_flight(self):
        beta1, beta2 = 1.5, 3.0
        sigma_u1 = (np.math.gamma(1 + beta1) * np.sin(np.pi * beta1 / 2) /
                   (np.math.gamma((1 + beta1) / 2) * beta1 * 2 ** ((beta1 - 1) / 2))) ** (1 / beta1)
        sigma_u2 = (np.math.gamma(1 + beta2) * np.sin(np.pi * beta2 / 2) /
                   (np.math.gamma((1 + beta2) / 2) * beta2 * 2 ** ((beta2 - 1) / 2))) ** (1 / beta2)
        u1, u2 = np.random.normal(0, sigma_u1), np.random.normal(0, sigma_u2)
        v = np.random.normal(0, 1)
        step1 = u1 / (abs(v) ** (1 / beta1))
        step2 = u2 / (abs(v) ** (1 / beta2))
        return step1, step2

    def adapt_parameters(self):
        self.F = 0.4 + 0.6 * np.random.rand()  # Dynamic mutation factor for varied search behavior
        self.CR = 0.5 + 0.5 * np.random.rand()  # Flexible crossover range for adaptability

    def enhanced_mutation(self, best, idx):
        if np.random.rand() < 0.7:
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

                # Possible Dual Lévy flight step
                levy_step1, levy_step2 = self.dual_levy_flight()
                trial = trial + levy_step1 * (trial - best) + levy_step2 * np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Selection
                trial_score = evaluate(trial)
                if trial_score is not None and trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

        return best, best_score