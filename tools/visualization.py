import matplotlib.pyplot as plt

def visual(dataset, y):
    plt.scatter(dataset.x, dataset.y, c='r', label='Ground Truth')
    plt.scatter(dataset.x, y, c='b', label='Prediction')
    plt.legend()
    plt.show()