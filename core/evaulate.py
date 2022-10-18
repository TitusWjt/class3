from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch

def evaluate(labels, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return acc, nmi, ari,  pur




def evaluation(Pretrain_p, model, dataset, view, data_size, class_num, device):
    with torch.no_grad():
        model.eval()
        test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
        Zs = []
        Rs = []
        for v in range(view):
            Zs.append([])
            Rs.append([])
        labels_vector = []
        for step, (xs, y, idx) in enumerate(test_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
                zs, rs = model(xs)
            for v in range(view):
                zs[v] = zs[v].detach()
                rs[v] = rs[v].detach()
                Zs[v].extend(zs[v].cpu().detach().numpy())
                Rs[v].extend(rs[v].cpu().detach().numpy())
            labels_vector.extend(y.numpy())
        labels_vector = np.array(labels_vector).reshape(data_size)
        for v in range(view):
            Zs[v] = np.array(Zs[v])
            Rs[v] = np.array(Rs[v])

        print("Clustering results on common features of each view:")
        for v in range(view):
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(Zs[v])
            scores = evaluate(labels_vector, y_pred)
            print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
                                                                                     v + 1, nmi,
                                                                                     v + 1, ari,
                                                                                     v + 1, pur))

    return scores