import numpy as np

class EnhancedHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.8  # Initial DE Mutation factor
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.scores = np.full(self.initial_population_size, np.inf)
        self.func_evals = 0
        self.dynamic_population_size = self.initial_population_size

    def levy_flight(self, L):
        u = np.random.normal(0, 1) * (L ** (-1/3))
        v = np.random.normal(0, 1)
        step = u / (abs(v) ** (1/3))
        return step

    def dynamic_mutation_factor(self, current_eval):
        return self.F * (1 - current_eval / self.budget)

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

        for i in range(self.dynamic_population_size):
            self.scores[i] = evaluate(self.population[i])

        while self.func_evals < self.budget:
            self.dynamic_population_size = max(5, int(self.initial_population_size * (1 - self.func_evals / self.budget)))

            for i in range(self.dynamic_population_size):
                indices = list(range(self.dynamic_population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Dynamic Mutation
                F_dynamic = self.dynamic_mutation_factor(self.func_evals)
                mutant = self.population[a] + F_dynamic * (self.population[b] - self.population[c])

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