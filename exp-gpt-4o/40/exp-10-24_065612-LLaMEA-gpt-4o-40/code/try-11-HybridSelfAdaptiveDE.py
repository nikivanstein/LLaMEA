import numpy as np

class HybridSelfAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size_initial = 12 * dim
        self.pop_size = self.pop_size_initial
        self.F = 0.75
        self.CR = 0.85
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.pop_size_initial, dim))
        self.local_search_probs = [0.1, 0.05]  # Probabilities for different local searches

    def _mutate(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c, d = np.random.choice(indices, 4, replace=False)
        mutant = np.clip(self.population[a] + self.F * (self.population[b] - self.population[c]) + 0.5 * (self.population[d] - self.population[idx]),
                         self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _adaptive_CR(self, generation_count):
        return 0.75 - 0.5 * (generation_count / (self.budget/self.pop_size_initial))

    def _adaptive_F(self, generation_count):
        return 0.3 + 0.6 * (generation_count / (self.budget/self.pop_size_initial))

    def _adjust_population_size(self, generation_count):
        factor = 1 - (generation_count / (self.budget/self.pop_size_initial))
        self.pop_size = int(max(5, self.pop_size_initial * factor))

    def _local_search(self, individual, method):
        if method == 0:
            perturbation = np.random.normal(0, 0.1, self.dim)
        else:
            perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        return np.clip(individual + perturbation, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        eval_count = 0
        best_sol = None
        best_val = float('inf')
        
        generation_count = 0

        while eval_count < self.budget:
            self._adjust_population_size(generation_count)
            new_population = np.copy(self.population[:self.pop_size])

            for i in range(self.pop_size):
                self.F = self._adaptive_F(generation_count)
                self.CR = self._adaptive_CR(generation_count)

                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant)
                
                if np.random.rand() < np.max(self.local_search_probs):
                    method = np.random.choice([0, 1], p=self.local_search_probs)
                    trial = self._local_search(trial, method)
                
                trial_val = func(trial)
                eval_count += 1

                if trial_val < func(self.population[i]):
                    new_population[i] = trial
                    if trial_val < best_val:
                        best_val = trial_val
                        best_sol = trial

                if eval_count >= self.budget:
                    break
            
            self.population[:self.pop_size] = new_population
            generation_count += 1

        return best_sol