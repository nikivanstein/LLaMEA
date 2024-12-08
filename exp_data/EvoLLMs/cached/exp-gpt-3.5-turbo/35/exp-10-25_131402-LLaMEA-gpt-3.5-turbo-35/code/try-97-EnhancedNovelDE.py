import numpy as np

class EnhancedNovelDE(NovelDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.prob_refinement = 0.35
