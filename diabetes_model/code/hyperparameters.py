from hyperopt import hp
from hyperopt.pyll.base import scope

space = {
    "eta": hp.uniform("eta", 0, 0.4),
    "max_depth": scope.int(hp.quniform("max_depth", 1, 8, 1)),
    "subsample": hp.uniform("subsample", 0.6, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.9, 1),
    "n_estimators": scope.int(hp.quniform("n_estimators", 20, 2000, 10)),
    "gamma": hp.uniform("gamma", 0.0, 0.3),
    "lambda": hp.uniform("lambda", 0.0, 1.0),
    "min_child_weight": hp.uniform("min_child_weight", 0.5, 0.8),
    "nthread": 4,
}
