import numpy as np

class EnhancedHybridDELevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20  # Initial population size
        self.population_growth_rate = 1.1  # Dynamic population growth rate
        self.max_population_size = 50  # Maximum allowed population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.7  # DE Mutation factor (adjusted)
        self.CR = 0.85  # Crossover probability (adjusted)
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.scores = np.full(self.initial_population_size, np.inf)
        self.func_evals = 0

    def levy_flight(self, L):
        u = np.random.normal(0, 1) * (L ** (-1/3))
        v = np.random.normal(0, 1)
        step = u / (abs(v) ** (1/3))
        energy_boost = np.random.uniform(0.9, 1.1)  # Energy boost factor
        return step * energy_boost

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

        for i in range(len(self.population)):
            self.scores[i] = evaluate(self.population[i])

        while self.func_evals < self.budget:
            population_size = min(int(len(self.population) * self.population_growth_rate), self.max_population_size)
            new_population = []
            new_scores = []

            for i in range(len(self.population)):
                indices = list(range(len(self.population)))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Mutation
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])

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
                    new_population.append(trial)
                    new_scores.append(trial_score)
                else:
                    new_population.append(self.population[i])
                    new_scores.append(self.scores[i])

            self.population = np.array(new_population)
            self.scores = np.array(new_scores)

        return best, best_score