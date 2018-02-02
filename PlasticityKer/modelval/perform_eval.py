"""
    Evaluate performance and making figures
"""
import numpy as np

def R2_corr(predictions, targets, x_fit):
    R2 = 1 - np.sum(np.square(predictions - targets)) / np.sum(np.square(targets - np.mean(targets)))
    # Calculate correlation coefficient
    corr = np.sum(np.dot((predictions - np.mean(predictions)).T, targets - np.mean(targets))) / np.std(
        predictions) / np.std(targets) / len(targets)
    b1 = np.sum(np.dot((predictions - np.mean(predictions)).T, targets - np.mean(targets))) / np.sum(
        np.square(targets - np.mean(targets)))
    b2 = np.mean(predictions) - np.mean(targets) * b1
    # x_fit = np.linspace(np.min(targets) - 1, np.max(targets) + 1, 100)
    y_fit = x_fit * b1 + b2
    # plt.plot(x_fit, y_fit, 'k--')
    # plt.plot(targets, predictions, 'ro')
    # plt.xlabel('targets')
    # plt.ylabel('predictions')
    # plt.title(''.join(('R2=%.4f' % (R2), ', Corr=%.4f' % (corr))))
    return R2, corr, y_fit

