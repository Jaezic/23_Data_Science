from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_squared_error, accuracy_score, precision_score, recall_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Classification Evaluation


def evaluate(args, model, predict, y):
    # print(f'Evaluation on test set, \n',
    #       'Score: {:.4f}, '.format(
    #     model.score(train_dataset.x, train_dataset.y)))

    metrics = Metrics(predict, y)
    print(f'Evaluation on test set, \n',
          'Accuracy: {:.4f}, Recall: {:.4f}, Precision: {:.4f}, F1 Score: {:.4f}'.format(
              metrics.accuracy,
              metrics.recall,
              metrics.precision,
              metrics.f1))

    # Visualization Confusion Matrix
    confusion = confusion_matrix(predict, y)
    print('Confusion Matrix: \n', confusion)
    if args.visual:
        sns.heatmap(confusion, annot=True, fmt='d')
        plt.show()

    return metrics


class Metrics:
    def __init__(self, predict, y):
        self.predict = predict
        self.y = y
        self.accuracy = accuracy_score(predict, y)
        self.recall = recall_score(predict, y, average='macro')
        self.precision = precision_score(predict, y, average='macro')
        self.f1 = f1_score(predict, y, average='macro')
