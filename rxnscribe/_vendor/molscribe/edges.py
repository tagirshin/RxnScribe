import numpy as np

def get_edge_prediction(edge_prob):
    if not edge_prob:
        return [], []
    n = len(edge_prob)
    if n == 0:
        return [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(5):
                edge_prob[i][j][k] = (edge_prob[i][j][k] + edge_prob[j][i][k]) / 2
                edge_prob[j][i][k] = edge_prob[i][j][k]
            edge_prob[i][j][5] = (edge_prob[i][j][5] + edge_prob[j][i][6]) / 2
            edge_prob[i][j][6] = (edge_prob[i][j][6] + edge_prob[j][i][5]) / 2
            edge_prob[j][i][5] = edge_prob[i][j][6]
            edge_prob[j][i][6] = edge_prob[i][j][5]
    prediction = np.argmax(edge_prob, axis=2).tolist()
    score = np.max(edge_prob, axis=2).tolist()
    return prediction, score
