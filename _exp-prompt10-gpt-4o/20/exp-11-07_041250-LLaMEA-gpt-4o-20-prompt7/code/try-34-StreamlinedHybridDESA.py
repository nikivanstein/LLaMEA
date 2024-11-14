import numpy as np

class StreamlinedHybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def _clip(self, x):
        return np.clip(x, self.lower_bound, self.upper_bound)
    
    def _initialize_population(self, pop_size):
        return np.random.uniform(self.lower_bound, self.upper_bound, (pop_size, self.dim))
    
    def _adaptive_parameters(self, eval_count):
        F = 0.8 - 0.3 * (eval_count / self.budget)  # Adjusted mutation factor
        Cr = 0.8 + 0.2 * np.cos(eval_count / self.budget * np.pi)  # Adjusted crossover probability
        return F, Cr
    
    def _differential_evolution(self, func, population, scores, eval_count):
        pop_size = len(population)
        F, Cr = self._adaptive_parameters(eval_count)
        for i in range(pop_size):
            indices = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            a, b, c = population[indices]
            mutant = a + F * (b - c)
            mutant = self._clip(mutant)
            
            crossover_mask = np.random.rand(self.dim) < Cr
            trial = np.where(crossover_mask, mutant, population[i])
            trial_score = func(trial)
            
            if trial_score < scores[i]:
                population[i], scores[i] = trial, trial_score
        return population, scores
    
    def _simulated_annealing(self, func, x, score, eval_count):
        temperature = 1.0
        cooling_rate = 0.9  # Simplified cooling rate
        iteration_limit = 10  # Static iteration limit for consistency
        for _ in range(iteration_limit):
            neighbor = x + np.random.normal(0, 0.1, self.dim)  # Adjusted perturbation scale
            neighbor = self._clip(neighbor)
            neighbor_score = func(neighbor)
            
            if neighbor_score < score or np.random.rand() < np.exp((score - neighbor_score) / temperature):
                x, score = neighbor, neighbor_score
            
            temperature *= cooling_rate
            eval_count += 1
        return x, score, eval_count
    
    def __call__(self, func):
        pop_size = min(25, self.budget // 6)  # Adjusted population size
        population = self._initialize_population(pop_size)
        scores = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        while eval_count < self.budget:
            population, scores = self._differential_evolution(func, population, scores, eval_count)
            best_idx = np.argmin(scores)
            best_individual, best_score = population[best_idx], scores[best_idx]
            
            best_individual, best_score, eval_count = self._simulated_annealing(func, best_individual, best_score, eval_count)
            
            population[best_idx], scores[best_idx] = best_individual, best_score
            
        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]