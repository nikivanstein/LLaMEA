import numpy as np
from sklearn.cluster import KMeans

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 8 * dim  # Reduced population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.pop_size
        
        while evals < self.budget:
            adapt_factor = 1 + evals / self.budget  # Increase diversity over time
            kmeans = KMeans(n_clusters=min(max(2, evals // (self.budget // 5)), self.pop_size))
            labels = kmeans.fit_predict(pop)
            for i in range(self.pop_size):
                cluster_indices = np.where(labels == labels[i])[0]
                if len(cluster_indices) > 1:
                    idxs = [idx for idx in cluster_indices if idx != i]
                    a, b = pop[np.random.choice(idxs, 2, replace=False)]
                else:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b = pop[np.random.choice(idxs, 2, replace=False)]
                F = 0.6 + np.random.rand() * 0.4
                CR = 0.7 + np.random.rand() * 0.3
                mutant = np.clip(a + F * (b - pop[i]), self.lower_bound, self.upper_bound)
                
                trial = np.array([mutant[j] if np.random.rand() < CR else pop[i][j] for j in range(self.dim)])
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                
                if evals < self.budget:
                    perturbation_size = 0.1 * (1.0 - evals / self.budget)
                    grad_perturb = np.random.randn(self.dim) * perturbation_size * adapt_factor
                    local_trial = pop[i] + grad_perturb
                    local_trial = np.clip(local_trial, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_trial)
                    evals += 1
                    
                    if local_fitness < fitness[i]:
                        pop[i] = local_trial
                        fitness[i] = local_fitness
            
            if evals >= self.budget:
                break
        
        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]