from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_squared_error, accuracy_score, precision_score, recall_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Classification Evaluation
def evaluate(model, predict, y):
    # print(f'Evaluation on test set, \n',
    #       'Score: {:.4f}, '.format(
    #     model.score(train_dataset.x, train_dataset.y)))

    ClassificationReport = classification_report(predict, y)
    print(f'Evaluation on test set, \n',
          'Accuracy: {:.4f}, Recall: {:.4f}, Precision: {:.4f}, F1 Score: {:.4f}'.format(
              accuracy_score(predict, y),
              recall_score(predict, y, average='macro'),
              precision_score(predict, y, average='macro'),
              f1_score(predict, y, average='macro')))

    # Visualization Confusion Matrix
    confusion = confusion_matrix(predict, y)
    print('Confusion Matrix: \n', confusion)
    sns.heatmap(confusion, annot=True, fmt='d')
    plt.show()

