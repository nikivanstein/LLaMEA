from scipy.stats import logistic

class EnhancedDynamicMutationPSO_DE_Optimizer(DynamicMutationPSO_DE_Optimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        
    def __call__(self, func):
        def chaotic_map(dim):
            x = np.random.uniform(-5.0, 5.0, dim)
            for _ in range(10):
                x = logistic.cdf(4 * x - 2)
            return x
        
        population = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocities = np.array([chaotic_map(self.dim) for _ in range(self.swarm_size)])
        pbest = population.copy()
        
        # Remaining code unchanged
        
        return gbest