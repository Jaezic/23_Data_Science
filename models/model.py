from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import os

def build_model(args):
    """
    Build model in Arguments, and load parameters if args.param_load is True
        Args:
            args: arguments from argument_parser()
        
        Returns:
            model: instance of model
    """
    
    p = {}
    if args.model != 'voting':
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
    elif args.model == 'bag':
        model = BaggingClassifier(base_estimator=DecisionTreeClassifier(
            max_depth=1), random_state=args.seed, **p)
    elif args.model == 'kmeans':
        model = KMeans(n_clusters=args.num_class, random_state=args.seed)
    elif args.model == 'voting':
        models = []
        if args.voting_list == None or len(args.voting_list) == 0:
            raise ValueError('Empty voting list')
        for model in args.voting_list:
            args.model = model
            models.append((args.model, build_model(args)))
        args.model = 'voting'
        p = load_param(args)
        model = VotingClassifier(estimators=models, **p)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    
    # should not print model here. but for debug, print it.
    print(model)
    print('-' * 60)
    return model


def load_param(args):
    """
    Load hyperparameters from file
        Args:
            args: arguments from argument_parser()
            
        Returns:
            parmas: hyperparameters dict
    """
    if args.param_load == False:
        return {}
    try:
        with open(os.path.join(args.param_path, args.model+'_tune.txt'), 'r') as f:
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