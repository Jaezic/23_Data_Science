from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import os
# dt(decision tree), lr(logistic regression), knn(k-nearest neighbors), rf(random forest), ab(AdaBoost), gb(gradient boosting)


def build_model(args):
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
        model = KMeans(n_clusters=args.target_number, random_state=args.seed)
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
    print(model)
    print('-' * 60)
    return model


def load_param(args):
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

# import random
# import pandas as pd
# from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from collections import Counter
# import numpy as np
# import os

# def build_model(args):
#     p = load_param(args)
#     if args.model == 'dt':
#         model = DecisionTreeClassifier(random_state=args.seed, **p)
#     elif args.model == 'lr':
#         model = LogisticRegression(random_state=args.seed, **p)
#     elif args.model == 'knn':
#         model = KNeighborsClassifier(**p)
#     elif args.model == 'rf':
#         model = RandomForestClassifier(random_state=args.seed, **p)
#     elif args.model == 'ab':
#         model = AdaBoostClassifier(random_state=args.seed, **p)
#     elif args.model == 'gb':
#         model = GradientBoostingClassifier(random_state=args.seed, **p)
#     else:
#         raise ValueError(f'Unknown model: {args.model}')

#     return model


# def load_param(args):
#     try:
#         with open(os.path.join(args.param_path,args.model+'_tune'), 'r') as f:
#             params = dict(eval(f.read()))
#             print('Loaded Hyperparameters')
#             print(params)
#     except FileNotFoundError:
#         print('Not Found Hyperparameters File')
#         params = {}
#     except SyntaxError:
#         print('Syntax Error Hyperparameters File')
#         params = {}
#     return params


# def bagging(datasets, num_bagging, sample_size, model_builder):
#     models = []

#     for _ in range(num_bagging):
#         sampled_datasets = random.choices(datasets, k=sample_size)

#         model = model_builder()
#         X_train, y_train = zip(*sampled_datasets)
#         model.fit(X_train, y_train)

#         models.append(model)

#     return models


# df = pd.read_csv('../dataset/preprocessed.csv')

# datasets = []
# for _, row in df.iterrows():
#     features = row[:-1].values.tolist()
#     label = row[-1]
#     datasets.append((features, label))

# num_bagging = 5
# sample_size = len(datasets)

# bagging_models = bagging(datasets, num_bagging, sample_size, build_model)

# new_data = [21, 37, 2.2, 3, 27.3, 3.4, 0, 0, 7, 1, 1, 269, 64.8]
# predictions = []
# for model in bagging_models:
#     prediction = model.predict([new_data])
#     predictions.append(prediction.tolist())

# prediction_counts = Counter(map(tuple, predictions))
# ensemble_prediction = prediction_counts.most_common(1)[0][0]
# print("Ensemble prediction:", ensemble_prediction)
