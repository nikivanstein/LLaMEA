from scipy.optimize import minimize

class ImprovedHybridPSOSA(HybridPSOSA):
    def __init__(self, budget, dim, swarm_size=30, max_iter=100):
        super().__init__(budget, dim, swarm_size, max_iter)
        self.local_search_iter = 20

    def local_search(self, initial_solution):
        res = minimize(func, initial_solution, method='Nelder-Mead', options={'maxiter': self.local_search_iter})
        return res.x

    def __call__(self, func):
        best_solution = pso_search()
        remaining_budget = self.budget - self.max_iter * self.swarm_size
        if remaining_budget > 0:
            best_solution = self.local_search(best_solution)

        return best_solution