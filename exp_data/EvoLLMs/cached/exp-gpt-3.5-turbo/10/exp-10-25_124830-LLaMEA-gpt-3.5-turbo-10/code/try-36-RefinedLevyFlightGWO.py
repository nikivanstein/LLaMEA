import numpy as np

class RefinedLevyFlightGWO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.max_iter = budget // (3 * dim)
        self.alpha = np.zeros(dim)
        self.beta = np.zeros(dim)
        self.delta = np.zeros(dim)
        self.pop_size = 5 + int(15 * np.sqrt(dim))
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def levy_flight():
            beta = 1.5
            sigma = (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma)
            v = np.random.normal(0, 1)
            step = u / abs(v) ** (1 / beta)
            return step * np.clip(1.0 / np.sqrt(self.dim), 0.01, 1.0)

        def update_alpha_beta_delta(wolves):
            sorted_wolves = sorted(wolves, key=lambda x: x['fitness'])
            self.alpha = sorted_wolves[0]['position']
            self.beta = sorted_wolves[1]['position']
            self.delta = sorted_wolves[2]['position']
            
        def boundary_check(position):
            return np.clip(position, self.lb, self.ub)
        
        wolves = [{'position': np.random.uniform(self.lb, self.ub, self.dim), 'fitness': np.inf} for _ in range(self.pop_size)]
        best_fitness = np.inf
        
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                wolves[i]['position'] = boundary_check(wolves[i]['position'])
                wolves[i]['fitness'] = func(wolves[i]['position'])
                best_fitness = min(best_fitness, wolves[i]['fitness'])
            
            update_alpha_beta_delta(wolves)
            
            for i in range(self.pop_size):
                a = 2 - 2 * (_ + 1) / self.max_iter
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * self.alpha - wolves[i]['position'])
                X1 = self.alpha - A * D
                
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * self.beta - wolves[i]['position'])
                X2 = self.beta - A * D
                
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * self.delta - wolves[i]['position'])
                X3 = self.delta - A * D
                
                wolves[i]['position'] = boundary_check((X1 + X2 + X3) / 3)
                
                step = levy_flight()
                wolves[i]['position'] += step * np.random.normal(0, 1, self.dim)
                wolves[i]['position'] = boundary_check(wolves[i]['position'])
        
        return best_fitness