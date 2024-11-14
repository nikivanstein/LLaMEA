import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def _clip(self, x):
        return np.clip(x, self.lower_bound, self.upper_bound)
    
    def _initialize_population(self, pop_size):
        return np.random.uniform(self.lower_bound, self.upper_bound, (pop_size, self.dim))
    
    def _differential_evolution(self, population, scores):
        pop_size = len(population)
        for i in range(pop_size):
            indices = list(range(pop_size))
            indices.remove(i)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = a + 0.8 * (b - c)
            mutant = self._clip(mutant)
            
            crossover_mask = np.random.rand(self.dim) < 0.9
            trial = np.where(crossover_mask, mutant, population[i])
            trial_score = func(trial)
            
            if trial_score < scores[i]:
                population[i] = trial
                scores[i] = trial_score
        return population, scores
    
    def _simulated_annealing(self, x, score):
        temperature = 1.0
        cooling_rate = 0.99
        for _ in range(100):
            neighbor = x + np.random.normal(scale=0.1, size=self.dim)
            neighbor = self._clip(neighbor)
            neighbor_score = func(neighbor)
            
            if neighbor_score < score or np.random.rand() < np.exp((score - neighbor_score) / temperature):
                x, score = neighbor, neighbor_score
            
            temperature *= cooling_rate
        return x, score
    
    def __call__(self, func):
        pop_size = 20
        population = self._initialize_population(pop_size)
        scores = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        while eval_count < self.budget:
            population, scores = self._differential_evolution(population, scores)
            best_idx = np.argmin(scores)
            best_individual, best_score = population[best_idx], scores[best_idx]
            
            best_individual, best_score = self._simulated_annealing(best_individual, best_score)
            
            population[best_idx] = best_individual
            scores[best_idx] = best_score
            eval_count += 100  # Simulated Annealing uses constant evaluations
            
        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]