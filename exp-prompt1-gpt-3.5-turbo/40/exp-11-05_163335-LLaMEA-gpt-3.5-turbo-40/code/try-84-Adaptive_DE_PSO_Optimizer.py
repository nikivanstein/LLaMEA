class Adaptive_DE_PSO_Optimizer:
    def __init__(self, budget, dim, npop=30, F=0.5, CR=0.9, w=0.7, c1=1.5, c2=1.5, F_decay=0.95, CR_decay=0.95, w_decay=0.99, c1_decay=0.99, c2_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.npop = npop
        self.F = F
        self.CR = CR
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.F_decay = F_decay
        self.CR_decay = CR_decay
        self.w_decay = w_decay
        self.c1_decay = c1_decay
        self.c2_decay = c2_decay

    def __call__(self, func):
        def evaluate(pop):
            return np.array([func(ind) for ind in pop])

        def initialize_population(npop, dim):
            return np.array([[chaotic_map(np.random.rand()) * 10.0 - 5.0 for _ in range(dim)] for _ in range(npop)])

        def adapt_parameters(iter, max_iter):
            self.F = self.F * self.F_decay ** (iter / max_iter)
            self.CR = self.CR * self.CR_decay ** (iter / max_iter)
            self.w = self.w * self.w_decay ** (iter / max_iter)
            self.c1 = self.c1 * self.c1_decay ** (iter / max_iter)
            self.c2 = self.c2 * self.c2_decay ** (iter / max_iter)

        pop = initialize_population(self.npop, self.dim)
        pbest = np.copy(pop)
        gbest = pop[np.argmin(evaluate(pop))]
        velocity = np.zeros_like(pop)
        
        for i in range(self.budget):
            adapt_parameters(i, self.budget)
            
            pop, pbest, gbest = mutate(pbest, gbest, pop, self.F, self.CR)
            velocity = update_velocity(pbest, pop, velocity, gbest, self.w, self.c1, self.c2)
            pop += velocity
        
        return gbest