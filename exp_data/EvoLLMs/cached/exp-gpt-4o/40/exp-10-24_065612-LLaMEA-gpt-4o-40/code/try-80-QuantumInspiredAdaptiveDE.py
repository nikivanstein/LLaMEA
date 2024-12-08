import numpy as np

class QuantumInspiredAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size_initial = 10 * dim  # Slightly increased initial population size for diversity
        self.pop_size = self.pop_size_initial
        self.F = 0.7  # Modified F for balanced exploration and exploitation
        self.CR = 0.8  # Lower CR for more concentrated crossover
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.pop_size_initial, dim))
        self.local_search_prob = 0.5  # Higher probability for local search to refine solutions
        self.q_perturbation_mag = 0.1  # Quantum perturbation magnitude for local search
        self.noise_resistant_strategy = True  # Switch to toggle noise handling strategy

    def _mutate(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = np.clip(self.population[a] + self.F * (self.population[b] - self.population[c]), self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _adaptive_CR(self, generation_count):
        return 0.85 - 0.3 * (generation_count / (self.budget / self.pop_size_initial))

    def _adaptive_F(self, generation_count):
        return 0.5 + 0.4 * (generation_count / (self.budget / self.pop_size_initial))

    def _adjust_population_size(self, generation_count):
        decay_factor = 0.98  # Slower decay for maintaining diversity
        self.pop_size = int(max(5, self.pop_size_initial * decay_factor ** generation_count))

    def _quantum_local_search(self, individual):
        perturbation = np.random.uniform(-self.q_perturbation_mag, self.q_perturbation_mag, self.dim) * np.random.normal(0, 0.05)
        return np.clip(individual + perturbation, self.lower_bound, self.upper_bound)

    def _noise_resistant_evaluation(self, individual, func):
        evaluations = [func(individual + np.random.normal(0, 0.01, self.dim)) for _ in range(3)]
        return np.median(evaluations)

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
                
                if np.random.rand() < self.local_search_prob:
                    trial = self._quantum_local_search(trial)

                if self.noise_resistant_strategy:
                    trial_val = self._noise_resistant_evaluation(trial, func)
                else:
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