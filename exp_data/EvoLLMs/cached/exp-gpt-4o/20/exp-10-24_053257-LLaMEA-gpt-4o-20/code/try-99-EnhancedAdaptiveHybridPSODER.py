import numpy as np

class EnhancedAdaptiveHybridPSODER:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(60, budget // (dim * 3))
        self.w = 0.4 + 0.3 * np.random.rand()
        self.c1 = 1.4 + 0.2 * np.random.rand()
        self.c2 = 1.8 + 0.1 * np.random.rand()
        self.F = 0.5 + 0.2 * np.random.rand()
        self.CR = 0.7 + 0.2 * np.random.rand()
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        adapt_strategy = np.random.choice(['rank_based', 'elitist_recombination'])
        
        while self.evaluations < self.budget:
            scores = np.array([func(ind) for ind in self.population])
            self.evaluations += len(self.population)
            ordering = np.argsort(scores)
            self.personal_best_positions = self.population[ordering[:self.population_size // 2]]
            self.personal_best_scores = scores[ordering[:self.population_size // 2]]
            if min(scores) < self.global_best_score:
                self.global_best_score = min(scores)
                self.global_best_position = self.population[np.argmin(scores)]
            
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i % (self.population_size // 2)] - self.population[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

            if adapt_strategy == 'rank_based':
                self.c1, self.c2 = sorted(np.random.normal(loc=[1.5, 1.7], scale=0.1, size=2))
                self.w = 0.5 + 0.5 * (self.global_best_score / np.sum(self.personal_best_scores))
            elif adapt_strategy == 'elitist_recombination':
                for i in range(self.population_size):
                    indices = list(range(0, i)) + list(range(i + 1, self.population_size))
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant_vector = self.population[a] + self.F * (self.population[b] - self.population[c])
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                    trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.population[i])
                    trial_score = func(trial_vector)
                    self.evaluations += 1
                    if trial_score < scores[i]:
                        self.population[i] = trial_vector

        return self.global_best_position, self.global_best_score