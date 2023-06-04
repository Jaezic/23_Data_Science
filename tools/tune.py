import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tune(args, model, train_dataset):
    params = load_param_range(args)
    if args.tune == 'grid':
        search = GridSearchCV(model, params, cv=args.cv)
    elif args.tune == 'random':
        search = RandomizedSearchCV(
            model, params, cv=args.cv, n_iter=args.n_iter, random_state=args.seed)
    else:
        raise ValueError('Unknown hyperparameters tuning method.')

    if args.pca:
        train_dataset.PCA_pipeline(args, train_dataset, None)
    search.fit(train_dataset.x, train_dataset.y)

    # print SearchCV name
    print(search.__class__.__name__)
    print('Best Parameter\n ', search.best_params_)
    print(' Best Score :', search.best_score_)
    save_param(args, search.best_params_)


def load_param_range(args):
    try:
        with open(os.path.join(args.param_path, args.model+'_range.txt'), 'r') as f:
            params = dict(eval(f.read()))
            print('Tunning - Loaded Hyperparameters Range')
            print(params)
    except FileNotFoundError:
        raise ValueError('Tunning - Not Found Hyperparameters Range File')
    except SyntaxError:
        raise ValueError('Tunning - Syntax Error Hyperparameters Range File')
    print('-' * 60)
    return params


def save_param(args, dict):
    try:
        with open(os.path.join(args.param_path, args.model+'_tune.txt'), 'w') as f:
            f.write(str(dict))
            print(
                'Tunning - Saved Hyperparameters [' + args.param_path + ']')
    except SyntaxError:
        print('Tunning - Syntax Error Hyperparameters Range File')
