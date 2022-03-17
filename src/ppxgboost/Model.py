
# An XGBoost model is a collection of TreeNodes

class XGBoostModel:

    def __init__(self, trees):
        self.trees = list(trees)

    def get_features(self):
        feature_set = set()
        for t in self.trees:
            feature_set = feature_set.union(t.get_features())
        return feature_set

    # this probably isn't the right API, but it works for now
    def update_extreme_values(self, min_max):
        forest_min, forest_max = self.get_extreme_values()
        return {'min': min(min_max['min'], forest_min), 'max': max(min_max['max'], forest_max)}

    def get_extreme_values(self):
        min_val = float('inf')
        max_val = float('-inf')

        for t in self.trees:
            t_min, t_max = t.get_extreme_values()
            min_val = min(min_val, t_min)
            max_val = max(max_val, t_max)

        return min_val, max_val
