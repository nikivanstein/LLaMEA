class EnhancedAdaptiveDEImprovedOppositionRefined(EnhancedAdaptiveDEImprovedOpposition):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.NP = max(5, int(np.ceil(10 + 2 * np.sqrt(dim))))  # Dynamic population size adjustment

    def __call__(self, func):
        return super().__call__(func)