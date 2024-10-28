import numpy as np

class QuantumInspiredDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size_initial = 8 * dim
        self.pop_size = self.pop_size_initial
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.pop_size_initial, dim))
        self.local_search_prob = 0.4
        self.global_perturbation_prob = 0.2

    def _mutate(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = self._adaptive_F()
        mutant = np.clip(self.population[a] + F * (self.population[b] - self.population[c]), self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        CR = self._adaptive_CR()
        crossover_mask = np.random.rand(self.dim) < CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _adaptive_CR(self):
        return np.random.uniform(0.5, 0.9)
    
    def _adaptive_F(self):
        return np.random.uniform(0.5, 0.9)

    def _adjust_population_size(self, generation_count):
        decay_factor = 0.93
        self.pop_size = int(max(5, self.pop_size_initial * decay_factor ** generation_count))

    def _quantum_local_search(self, individual):
        perturbation = np.random.uniform(-0.2, 0.2, self.dim) * np.random.exponential(0.1)
        return np.clip(individual + perturbation, self.lower_bound, self.upper_bound)

    def _global_perturbation(self, individual):
        if np.random.rand() < self.global_perturbation_prob:
            perturbation = np.random.normal(0, 1.0, self.dim)
            return np.clip(individual + perturbation, self.lower_bound, self.upper_bound)
        return individual

    def __call__(self, func):
        eval_count = 0
        best_sol = None
        best_val = float('inf')
        
        generation_count = 0

        while eval_count < self.budget:
            self._adjust_population_size(generation_count)
            new_population = np.copy(self.population[:self.pop_size])

            for i in range(self.pop_size):
                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant)
                
                if np.random.rand() < self.local_search_prob:
                    trial = self._quantum_local_search(trial)

                trial = self._global_perturbation(trial)
                
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