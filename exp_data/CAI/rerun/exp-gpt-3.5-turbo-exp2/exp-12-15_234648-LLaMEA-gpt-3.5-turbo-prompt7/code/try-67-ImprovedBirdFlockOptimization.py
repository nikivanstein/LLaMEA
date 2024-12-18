class ImprovedBirdFlockOptimization(BirdFlockOptimization):
    def __call__(self, func):
        def update_velocity(velocity, position, global_best_pos, personal_best_pos, iteration):
            r1, r2 = np.random.rand(), np.random.rand()
            w = self.w * (1.0 - iteration / self.budget) * (1.0 - np.clip(fitness(global_best_pos) - fitness(personal_best_pos), 0, 1))  # Adaptive inertia weight adjustment
            return w * velocity + self.c1 * r1 * (personal_best_pos - position) + self.c2 * r2 * (global_best_pos - position)

        return super().__call__(func)