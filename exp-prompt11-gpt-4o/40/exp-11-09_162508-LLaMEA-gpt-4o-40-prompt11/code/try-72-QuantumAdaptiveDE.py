import numpy as np

class QuantumAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40  # Increased population size for better exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5  # Base mutation factor
        self.CR = 0.9  # Base crossover probability
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.func_evals = 0

    def levy_flight(self, L):
        beta = 1.5  # Adjusted to explore different step sizes
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, 1)
        step = u / (abs(v) ** (1 / beta))
        return step

    def quantum_search(self, current, best):
        return current + 0.01 * np.random.randn(self.dim) * (best - current)  # Quantum-inspired local search

    def adapt_parameters(self):
        if self.func_evals < self.budget * 0.5:
            self.F = 0.5 + 0.3 * np.random.rand()  # Higher variation in early phase
            self.CR = 0.8 + 0.2 * np.random.rand()  # Adjusted for early exploration
        else:
            self.F = 0.3 + 0.2 * np.random.rand()  # Tighter range in later phase
            self.CR = 0.9  # Focus on exploitation

    def dynamic_strategy_mutation(self, best, idx):
        if np.random.rand() < 0.5:
            indices = list(range(self.population_size))
            indices.remove(idx)
            a, b, c = np.random.choice(indices, 3, replace=False)
            return self.population[a] + self.F * (self.population[b] - self.population[c])
        else:
            indices = list(range(self.population_size))
            indices.remove(idx)
            best_indices = np.argsort(self.scores)[:3]  # Focus on top 3 solutions
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
                mutant = self.dynamic_strategy_mutation(best, i)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Quantum-inspired search addition
                quantum_trial = self.quantum_search(trial, best)
                trial = np.clip(quantum_trial, self.lower_bound, self.upper_bound)

                # Selection
                trial_score = evaluate(trial)
                if trial_score is not None and trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

        return best, best_score