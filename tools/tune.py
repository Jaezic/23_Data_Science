import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def tune(args, model, train_dataset):
    params = load_param_range(args)
    if args.tune == 'grid':
        search = GridSearchCV(model, params, cv=args.cv)
    elif args.tune == 'random':
        search = RandomizedSearchCV(model, params, cv=args.cv)
    else:
        raise ValueError('Unknown hyperparameters tuning method.')

    search.fit(train_dataset.x, train_dataset.y)

    # print SearchCV name
    print(search.__class__.__name__)
    print('Best Parameter\n ', search.best_params_)
    print(' Best Score :', search.best_score_)


def load_param_range(args):
    try:
        with open(os.path.join(args.param_path, args.model+'_range'), 'r') as f:
            params = dict(eval(f.read()))
            print('Tunning - Loaded Hyperparameters Range')
            print(params)
    except FileNotFoundError:
        print('Tunning - Not Found Hyperparameters Range File')
        params = {}
    except SyntaxError:
        print('Tunning - Syntax Error Hyperparameters Range File')
        params = {}
    print('-' * 60)
    return params
