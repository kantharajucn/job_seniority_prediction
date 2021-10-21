from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score
from functools import partial
from utils import get_train_data

space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
         'gamma': hp.uniform('gamma', 1, 9),
         'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
         'reg_lambda': hp.uniform('reg_lambda', 0, 1),
         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
         'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
         'n_estimators': 100,
         'objective': 'multi:softprob',
         'seed': 0
         }


def objective(space, model, train_data, pre_processing_fun):
    clf = model(
        objective=space['objective'],
        n_estimators=space['n_estimators'],
        max_depth=int(space['max_depth']),
        gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']),
        min_child_weight=int(space['min_child_weight']),
        colsample_bytree=int(space['colsample_bytree']))
    X_train, X_valid, y_train, y_valid = get_train_data(train_data, 0)
    X_train, y_train = pre_processing_fun(X_train, y_train)
    X_valid, y_valid = pre_processing_fun(X_valid, y_valid)
    evaluation = [(X_train, y_train), (X_valid, y_valid)]

    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10, verbose=False)

    pred = clf.predict(X_valid)
    accuracy = accuracy_score(y_valid, pred > 0.5)
    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}


def tune(model, train_data, pre_processing_fun):
    trials = Trials()
    fmin_objective = partial(objective,
                             model=model,
                             train_data=train_data,
                             pre_processing_fun=pre_processing_fun)
    best_hyperparams = fmin(fn=fmin_objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=10,
                            trials=trials)
    return best_hyperparams
