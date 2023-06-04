from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# dt(decision tree), lr(logistic regression), knn(k-nearest neighbors), rf(random forest), ab(AdaBoost), gb(gradient boosting)


def build_model(args):
    if args.model == 'dt':
        model = DecisionTreeClassifier(random_state=args.seed)
    elif args.model == 'lr':
        model = LogisticRegression(random_state=args.seed)
    elif args.model == 'knn':
        model = KNeighborsClassifier()
    elif args.model == 'rf':
        model = RandomForestClassifier(random_state=args.seed)
    elif args.model == 'ab':
        model = AdaBoostClassifier(random_state=args.seed)
    elif args.model == 'gb':
        model = GradientBoostingClassifier(random_state=args.seed)
    else:
        raise ValueError(f'Unknown model: {args.model}')

    return model
