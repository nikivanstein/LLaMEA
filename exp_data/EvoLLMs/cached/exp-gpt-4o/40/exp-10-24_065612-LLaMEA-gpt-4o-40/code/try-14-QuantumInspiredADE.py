import numpy as np

class QuantumInspiredADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size_initial = 10 * dim
        self.pop_size = self.pop_size_initial
        self.QF = 0.9  # Quantum scale factor
        self.CR = 0.8  # Lower crossover rate for stability
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.pop_size_initial, dim))
        self.local_search_prob = 0.20  # Adjusted probability of applying local search

    def _mutate(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = np.clip(self.population[a] + self.QF * (self.population[b] - self.population[c]), self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _quantum_walk(self, individual):
        step_size = np.random.normal(0, 0.1, self.dim)  # Quantum step size
        return np.clip(individual + step_size, self.lower_bound, self.upper_bound)

    def _adaptive_CR(self, generation_count):
        return 0.8 - 0.6 * (generation_count / (self.budget/self.pop_size_initial))

    def _adaptive_QF(self, generation_count):
        return 0.4 + 0.5 * (generation_count / (self.budget/self.pop_size_initial))

    def _adjust_population_size(self, generation_count):
        factor = 1 - (generation_count / (self.budget/self.pop_size_initial))
        self.pop_size = int(max(5, self.pop_size_initial * factor))

    def _local_search(self, individual):
        if np.random.rand() < 0.5:  # Randomly choose between local search or quantum walk
            perturbation = np.random.normal(0, 0.3, self.dim)  # Enhanced perturbation
            return np.clip(individual + perturbation, self.lower_bound, self.upper_bound)
        else:
            return self._quantum_walk(individual)

    def __call__(self, func):
        eval_count = 0
        best_sol = None
        best_val = float('inf')
        
        generation_count = 0

        while eval_count < self.budget:
            self._adjust_population_size(generation_count)
            new_population = np.copy(self.population[:self.pop_size])

            for i in range(self.pop_size):
                self.QF = self._adaptive_QF(generation_count)
                self.CR = self._adaptive_CR(generation_count)

                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant)
                
                if np.random.rand() < self.local_search_prob:
                    trial = self._local_search(trial)
                
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