import numpy as np

class EnhancedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Increased population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_min = 0.5  # Dynamic scaling factor
        self.F_max = 0.9
        self.CR_min = 0.1  # Adaptive crossover probability
        self.CR_max = 0.9
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.func_evals = 0

    def adaptive_crossover(self, score):
        """Adapt the crossover rate based on current score."""
        return self.CR_min + (self.CR_max - self.CR_min) * ((score - min(self.scores)) / (max(self.scores) - min(self.scores) + 1e-9))

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
                    best = ind
                return score
            else:
                return None

        for i in range(self.population_size):
            self.scores[i] = evaluate(self.population[i])

        while self.func_evals < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Dynamic Mutation
                F = self.F_min + np.random.rand() * (self.F_max - self.F_min)
                mutant = self.population[a] + F * (self.population[b] - self.population[c])

                # Adaptive Crossover
                CR = self.adaptive_crossover(self.scores[i])
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Clipping and Evaluation
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                trial_score = evaluate(trial)
                if trial_score is not None and trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

        return best, best_score