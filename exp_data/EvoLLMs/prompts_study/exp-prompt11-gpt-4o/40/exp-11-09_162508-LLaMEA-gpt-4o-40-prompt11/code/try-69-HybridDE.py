import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40  # Increase population for diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.3  # Lower initial mutation factor for better stability
        self.CR = 0.8  # Crossover probability to balance exploration and exploitation
        self.alpha = 0.3  # Weight for history learning
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.func_evals = 0
        self.history = np.zeros((self.population_size, dim))  # History buffer for learning

    def adapt_parameters(self):
        self.F = 0.2 + 0.5 * np.random.rand()  # Broader mutation factor range
        self.CR = 0.6 + 0.4 * np.random.rand()  # Enhanced crossover variability

    def multi_strategy_mutation(self, best, idx):
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
                mutant = self.multi_strategy_mutation(best, i)

                # Crossover with dynamic learning
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])
                trial = trial + self.alpha * self.history[i]  # Learning from history

                # Selection
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                trial_score = evaluate(trial)
                if trial_score is not None and trial_score < self.scores[i]:
                    self.history[i] = trial - self.population[i]  # Update history
                    self.population[i] = trial
                    self.scores[i] = trial_score

        return best, best_score