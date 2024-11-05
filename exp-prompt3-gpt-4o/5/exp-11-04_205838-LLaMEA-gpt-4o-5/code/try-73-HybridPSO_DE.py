import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.F = 0.5
        self.CR = 0.9
        
    def __call__(self, func):
        np.random.seed(0)
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        V = np.random.uniform(-abs(self.ub-self.lb), abs(self.ub-self.lb), (self.pop_size, self.dim))
        p_best = X.copy()
        p_best_vals = np.array([func(x) for x in X])
        g_best = p_best[np.argmin(p_best_vals)]
        
        eval_count = len(X)
        
        while eval_count < self.budget:
            # Dynamic inertia weight adjustment
            w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)
            
            # Particle Swarm Optimization step
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                V[i] = w * V[i] + self.c1 * r1 * (p_best[i] - X[i]) + self.c2 * r2 * (g_best - X[i])
                X[i] = np.clip(X[i] + V[i], self.lb, self.ub)
            
            # Adaptive population size reduction
            if eval_count > self.budget * 0.5 and self.pop_size > 5:
                self.pop_size -= 1
                X = X[:self.pop_size]
                V = V[:self.pop_size]
                p_best = p_best[:self.pop_size]
                p_best_vals = p_best_vals[:self.pop_size]
            
            # Differential Evolution step with adaptive F and stochastic CR
            self.F = 0.4 + 0.3 * (np.exp(-5.0 * eval_count / self.budget))
            self.CR = np.random.uniform(0.7, 0.9)
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = X[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, X[i])
                
                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < p_best_vals[i]:
                    p_best[i] = trial
                    p_best_vals[i] = f_trial
                    if f_trial < func(g_best):
                        g_best = trial
            
            if eval_count >= self.budget:
                break
        
        return g_best