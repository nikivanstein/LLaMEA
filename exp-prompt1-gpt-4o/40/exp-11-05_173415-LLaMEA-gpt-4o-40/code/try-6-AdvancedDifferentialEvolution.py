import numpy as np

class AdvancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 15 * dim
        self.F = 0.6
        self.CR = 0.7
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.elite_rate = 0.1
        self.neighbors = 5
        self.stagnation_counter = 0
        self.max_stagnation = 40

    def mutate(self, idx):
        idxs = np.random.choice(np.delete(np.arange(self.pop_size), [idx]), 3, replace=False)
        a, b, c = self.population[idxs]
        return a + self.F * (b - c)
    
    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, trial, target_idx, func):
        trial_fitness = func(trial)
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.fitness[target_idx] = trial_fitness
            return True
        return False
    
    def restart_population(self):
        elite_count = int(self.pop_size * self.elite_rate)
        elite_indices = np.argsort(self.fitness)[:elite_count]
        elite_population = self.population[elite_indices]
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.population[:elite_count] = elite_population
        self.fitness = np.array([np.inf] * self.pop_size)
    
    def neighborhood_search(self, idx, func):
        best_fit = self.fitness[idx]
        best_pos = self.population[idx].copy()
        for _ in range(self.neighbors):
            neighbor = best_pos + np.random.normal(0, 0.1, self.dim)
            neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
            fit = func(neighbor)
            if fit < best_fit:
                best_fit = fit
                best_pos = neighbor
        return best_fit, best_pos
    
    def __call__(self, func):
        evaluations = 0
        self.fitness = np.array([func(ind) for ind in self.population])
        evaluations += self.pop_size
        
        while evaluations < self.budget:
            improved = False
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                
                mutant_vec = self.mutate(i)
                mutant_vec = np.clip(mutant_vec, self.lower_bound, self.upper_bound)
                trial_vec = self.crossover(self.population[i], mutant_vec)
                
                if self.select(trial_vec, i, func):
                    evaluations += 1
                    improved = True
                else:
                    evaluations += 1
                
                if evaluations >= self.budget:
                    break
            
            if improved:
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
                
            if self.stagnation_counter >= self.max_stagnation:
                self.restart_population()
                self.stagnation_counter = 0
                self.fitness = np.array([func(ind) for ind in self.population])
                evaluations += self.pop_size
            
            if evaluations % (self.pop_size * 10) == 0:
                success_rate = np.sum(self.fitness < np.median(self.fitness)) / self.pop_size
                self.F = np.clip(self.F * (1 + (0.15 if success_rate > 0.15 else -0.1)), 0.3, 0.9)
                self.CR = np.clip(self.CR * (1 + (0.15 if success_rate > 0.15 else -0.1)), 0.3, 0.9)
        
        for i in range(self.pop_size):
            if evaluations >= self.budget:
                break
            new_fit, new_pos = self.neighborhood_search(i, func)
            if new_fit < self.fitness[i]:
                self.population[i] = new_pos
                self.fitness[i] = new_fit
                evaluations += 1
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]