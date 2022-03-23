
# An XGBoost model is a collection of TreeNodes

import ppxgboost.Tree

class XGBoostModel:

    def __init__(self, trees):
        self.trees = list(trees)

    # this probably isn't the right API, but it works for now
    def update_extreme_values(self, min_max):
        forest_min, forest_max = self.get_extreme_values()
        return {'min': min(min_max['min'], forest_min), 'max': max(min_max['max'], forest_max)}

    def get_features(self):
        features = set()
        for t in self.trees:
            features = features.union(t.get_features())
        return features

    def get_extreme_values(self):
        min_val = float('inf')
        max_val = float('-inf')
        for t in self.trees:
            t_min, t_max = t.get_extreme_values()
            min_val = min(min_val, t_min)
            max_val = max(max_val, t_max)
        return min_val, max_val

    def discretize(self):
        for t in self.trees:
            t.discretize()
