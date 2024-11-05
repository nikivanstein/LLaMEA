import cma

class CMAESOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        es = cma.CMAEvolutionStrategy(x0=[0] * self.dim, sigma0=0.5, inopts={'bounds': [-5, 5]})
        while not es.stop() and es.sp.t < self.budget:
            solutions = es.ask()
            values = [func(x) for x in solutions]
            es.tell(solutions, values)
        return es.result[0]