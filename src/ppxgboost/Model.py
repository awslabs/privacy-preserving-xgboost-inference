
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
        def combo(set1, set2):
            return set1.union(set2)
        return self._combine_trees(ppxgboost.Tree.get_features, combo)

    def get_extreme_values(self):
        def combo(minmax1, minmax2):
            min1, max1 = minmax1
            min2, max2 = minmax2
            return min(min1, min2), max(max1, max2)
        return self._combine_trees(ppxgboost.Tree.get_extreme_values, combo)

    def discretize(self):
        self._combine_trees(ppxgboost.Tree.discretize, lambda x, y: x)

    # combine a tree function into a single value for the forest/model
    # internal function
    def _combine_trees(self, f, combo):
        acc = f(self.trees[0])
        for i in range(1, len(self.trees)):
            acc = combo(acc, f(self.trees[i]))
        return acc
