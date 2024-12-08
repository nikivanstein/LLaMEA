class DynamicAdjustPSO(DynamicPSO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        self.population_size = 23 if dim <= 10 else 22  # Dynamic population size adjustment during optimization
        return super().__call__(func)