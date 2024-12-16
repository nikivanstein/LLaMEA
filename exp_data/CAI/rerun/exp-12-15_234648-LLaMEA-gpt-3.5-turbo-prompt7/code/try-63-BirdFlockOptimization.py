class BirdFlockOptimization:
    def __init__(self, budget, dim, num_birds=20, w=0.5, c1=1.5, c2=1.5, adaptive_acceleration=True):
        self.budget = budget
        self.dim = dim
        self.num_birds = num_birds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.adaptive_acceleration = adaptive_acceleration

    def __call__(self, func):
        def update_velocity(velocity, position, global_best_pos, personal_best_pos, iteration):
            r1, r2 = np.random.rand(), np.random.rand()
            w = self.w * (1.0 - iteration / self.budget)
            if self.adaptive_acceleration:
                c1 = self.c1 * (1.0 - iteration / self.budget)
                c2 = self.c2 * (1.0 - iteration / self.budget)
            else:
                c1, c2 = self.c1, self.c2
            return w * velocity + c1 * r1 * (personal_best_pos - position) + c2 * r2 * (global_best_pos - position)