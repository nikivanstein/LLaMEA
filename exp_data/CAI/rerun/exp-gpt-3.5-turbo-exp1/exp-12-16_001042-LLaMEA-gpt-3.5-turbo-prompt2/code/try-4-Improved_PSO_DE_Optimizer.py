class Improved_PSO_DE_Optimizer(PSO_DE_Optimizer):
    def __init__(self, budget, dim, swarm_size=30, mutation_factor=0.5, crossover_prob=0.9, w_max=0.9, w_min=0.4, c1=1.5, c2=2.0):
        super().__init__(budget, dim, swarm_size, mutation_factor, crossover_prob, w_max, c1, c2)
        self.w_max = w_max
        self.w_min = w_min
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        pbest = population.copy()
        pbest_scores = np.array([func(p) for p in pbest])
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx]
        gbest_score = pbest_scores[gbest_idx]
        
        for t in range(1, self.budget+1):
            w = self.w_max - (self.w_max - self.w_min) * t / self.budget
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            population += velocities
            
            for i in range(self.swarm_size):
                new_sol = de(population[i], pbest[[i, (i+1)%self.swarm_size, (i+2)%self.swarm_size]], self.mutation_factor)
                new_score = func(new_sol)
                if new_score < pbest_scores[i]:
                    pbest[i] = new_sol
                    pbest_scores[i] = new_score
                    if new_score < gbest_score:
                        gbest = new_sol
                        gbest_score = new_score
        
        return gbest