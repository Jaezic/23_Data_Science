from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import os
# dt(decision tree), lr(logistic regression), knn(k-nearest neighbors), rf(random forest), ab(AdaBoost), gb(gradient boosting)


def build_model(args):
    if args.tune == None:
        p = load_param(args)
    if args.model == 'dt':
        model = DecisionTreeClassifier(random_state=args.seed, **p)
    elif args.model == 'lr':
        model = LogisticRegression(random_state=args.seed, **p)
    elif args.model == 'knn':
        model = KNeighborsClassifier(**p)
    elif args.model == 'rf':
        model = RandomForestClassifier(random_state=args.seed, **p)
    elif args.model == 'ab':
        model = AdaBoostClassifier(random_state=args.seed, **p)
    elif args.model == 'gb':
        model = GradientBoostingClassifier(random_state=args.seed, **p)
    else:
        raise ValueError(f'Unknown model: {args.model}')

    return model


def load_param(args):
    try:
        with open(os.path.join(args.param_path,args.model+'_tune'), 'r') as f:
            parmas = dict(eval(f.read()))
            print('Loaded Hyperparameters')
            print(parmas)
    except FileNotFoundError:
        print('Not Found Hyperparameters File')
        parmas = {}
    except SyntaxError:
        print('Syntax Error Hyperparameters File')
        parmas = {}
    return parmas
