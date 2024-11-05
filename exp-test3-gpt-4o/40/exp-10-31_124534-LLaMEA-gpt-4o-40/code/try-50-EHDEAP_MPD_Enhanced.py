import numpy as np

class EHDEAP_MPD_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.subpopulations = 5
        self.subpop_size = (12 * dim) // self.subpopulations  
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.9  # Adjusted mutation factor
        self.base_crossover_prob = 0.85  # Adjusted base crossover probability
        self.crossover_decay_rate = 0.15  # Adjusted crossover decay rate
        self.evaluations = 0
        self.elite_fraction = 0.35  # Adjusted elite fraction
        self.merging_interval = 120  # Adjusted merging interval
        self.global_best = None  # Track global best solution

    def __call__(self, func):
        populations = [self._initialize_population() for _ in range(self.subpopulations)]
        fitness = [np.array([func(ind) for ind in pop]) for pop in populations]
        self.evaluations += self.subpop_size * self.subpopulations
        
        self.global_best = self._get_global_best(populations, fitness)
        
        while self.evaluations < self.budget:
            for s, population in enumerate(populations):
                if self.evaluations >= self.budget:
                    break
                for i in range(self.subpop_size):
                    if self.evaluations >= self.budget:
                        break

                    elite_indices = np.argsort(fitness[s])[:int(self.elite_fraction * self.subpop_size)]
                    elite = population[np.random.choice(elite_indices)]
                    indices = list(range(self.subpop_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = self._hybrid_mutate(population[a], elite, population[c])

                    # Adaptive dynamic crossover
                    self.crossover_prob = self.base_crossover_prob - (self.crossover_decay_rate * (self.evaluations / self.budget))
                    trial = self._crossover(population[i], mutant)

                    # Enhanced random spatial restructuring
                    trial = self._random_restructuring(trial)

                    trial_fitness = func(trial)
                    self.evaluations += 1

                    if trial_fitness < fitness[s][i]:
                        population[i] = trial
                        fitness[s][i] = trial_fitness
                
                if self.evaluations % self.merging_interval == 0:
                    self._merge_subpopulations(populations, fitness)

        best_idx = np.argmin([f.min() for f in fitness])
        best_subpop = populations[best_idx]
        best_fit_idx = np.argmin(fitness[best_idx])
        return best_subpop[best_fit_idx]

    def _initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.subpop_size, self.dim))

    def _hybrid_mutate(self, a, b, c):
        rand_vector = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        mutant = np.clip(a + self.mutation_factor * (b - c) + 0.1 * (rand_vector - a), self.bounds[0], self.bounds[1])
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _random_restructuring(self, trial):
        if np.random.rand() < 0.1: 
            trial += np.random.normal(0, 0.1, self.dim)
        return np.clip(trial, self.bounds[0], self.bounds[1])
    
    def _get_global_best(self, populations, fitness):
        best_fit = np.inf
        best_sol = None
        for s, pop in enumerate(populations):
            min_idx = np.argmin(fitness[s])
            if fitness[s][min_idx] < best_fit:
                best_fit = fitness[s][min_idx]
                best_sol = pop[min_idx]
        return best_sol
    
    def _merge_subpopulations(self, populations, fitness):
        subpop_fitness = np.array([f.mean() for f in fitness])
        sorted_indices = np.argsort(subpop_fitness)
        half = len(sorted_indices) // 2
        for i in range(half, len(sorted_indices)):
            selected_idx = np.random.choice(sorted_indices[:half])
            populations[sorted_indices[i]] = populations[selected_idx]
            fitness[sorted_indices[i]] = fitness[selected_idx]