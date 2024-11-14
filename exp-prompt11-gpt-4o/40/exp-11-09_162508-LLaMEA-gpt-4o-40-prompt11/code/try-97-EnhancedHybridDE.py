import numpy as np

class EnhancedHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30  # Reduced initial population for faster adaptation
        self.population_size = int(self.initial_population_size * 1.5)  # Adjusted population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5  # More balanced mutation factor
        self.CR = 0.9  # Higher crossover probability to encourage diversity
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.func_evals = 0

    def levy_flight(self):
        beta = 1.3  # Adjusted Levy flight parameter for different step size
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, 1)
        step = u / (abs(v) ** (1 / beta))
        return step

    def adapt_parameters(self):
        self.F = 0.4 + 0.6 * np.random.rand()  # Wider range for mutation factor
        self.CR = 0.7 + 0.3 * np.random.rand()  # Adjusted crossover range

    def enhanced_mutation(self, best, idx):
        if np.random.rand() < 0.5:
            indices = list(range(self.population_size))
            indices.remove(idx)
            a, b, c = np.random.choice(indices, 3, replace=False)
            return best + self.F * (self.population[a] - self.population[b] + self.population[c] - self.population[idx])
        else:
            indices = list(range(self.population_size))
            indices.remove(idx)
            a, b = np.random.choice(indices, 2, replace=False)
            return self.population[a] + self.F * (best - self.population[b])

    def opposite_learning(self, indiv):
        return self.lower_bound + self.upper_bound - indiv

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

                # Opposite Learning Strategy
                opposite = self.opposite_learning(trial)
                opposite_score = evaluate(opposite)
                if opposite_score is not None and opposite_score < self.scores[i]:
                    trial = opposite
                    trial_score = opposite_score
                else:
                    trial_score = evaluate(trial)
                
                # Possible Lévy flight step
                if trial_score is not None and trial_score < self.scores[i]:
                    levy_step = self.levy_flight() * (trial - best)
                    trial = trial + levy_step
                    trial = np.clip(trial, self.lower_bound, self.upper_bound)
                    self.population[i] = trial
                    self.scores[i] = trial_score

                if trial_score is not None and trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

        return best, best_score