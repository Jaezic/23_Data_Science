from sklearn.metrics import mean_squared_error, accuracy_score


def evaluate(model, dataset, y):
    print(f'Evaluation on test set, \n',
          'Score: {:.4f}, MSE: {:.4f}'.format(
        model.score(dataset.x, dataset.y), mean_squared_error(dataset.y, y)))
