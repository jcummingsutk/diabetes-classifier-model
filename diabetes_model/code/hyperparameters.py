from hyperopt import hp
from hyperopt.pyll.base import scope

SPACE = {
    "eta": hp.uniform("eta", 0, 0.4),
    "max_depth": scope.int(hp.quniform("max_depth", 2, 5, 1)),
    "subsample": hp.uniform("subsample", 0.5, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    "n_estimators": scope.int(hp.quniform("n_estimators", 20, 2000, 10)),
    "lambda": hp.uniform("lambda", 0.4, 0.7),
    "gamma": hp.uniform("gamma", 0.1, 0.3),
    "min_child_weight": hp.uniform("min_child_weight", 0, 1),
    "nthread": 4,
}
