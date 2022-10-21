from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch

from core.evaulate_method import cluster_acc, purity, get_cluster_sols, get_y_preds, clustering_metric


def vaild(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return dict({'acc': acc, 'nmi': nmi, 'ari': ari, 'pur': pur})

def evaluate(x_list, y):
    """Get scores of clustering"""
    #为什么聚类中心的个数是这样得到的
    n_clusters = np.size(np.unique(y))
    x_final_concat = np.concatenate(x_list[:], axis=1)   #就一去中括号的作用
    kmeans_assignments, km = get_cluster_sols(x_final_concat, ClusterClass=KMeans, n_clusters=n_clusters,
                                              init_args={'n_init': 10})  #这个东西使y的值发生了变化
    y_preds = get_y_preds(y, kmeans_assignments, n_clusters)
    if np.min(y) == 1:
        y = y - 1
    scores, _ = clustering_metric(y, kmeans_assignments, n_clusters)
    ret = {}
    ret['kmeans'] = scores
    return ret


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
        if min(labels_vector) == 1:
            labels_vector = labels_vector - 1
        for v in range(view):
            Zs[v] = np.array(Zs[v])
            Rs[v] = np.array(Rs[v])

        # print("Clustering results on common features of each view:")
        # for v in range(view):
        #     kmeans = KMeans(n_clusters=class_num, n_init=100)
        #     y_pred = kmeans.fit_predict(Zs[v])
        #
        #     scores = vaild(labels_vector, y_pred)
        #     print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, scores['acc'],
        #                                                                              v + 1, scores['nmi'],
        #                                                                              v + 1, scores['ari'],
        #                                                                              v + 1, scores['pur']))
        print("Clustering results on common features of each view:")
        scores_each = []
        for v in range(view):
            scores_each.append([])
            scores = evaluate([Zs[v]], labels_vector)
            print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, scores['kmeans']['accuracy'],
                                                                                     v + 1, scores['kmeans']['NMI'],
                                                                                     v + 1, scores['kmeans']['ARI'],
                                                                                     v + 1, scores['kmeans']['pur']))
            scores_each[v].append(scores['kmeans']['accuracy'])
            scores_each[v].append(scores['kmeans']['NMI'])
            scores_each[v].append(scores['kmeans']['ARI'])
            scores_each[v].append(scores['kmeans']['pur'])


        print("Clustering results on common features of all view:")
        latent_fusion = np.concatenate(Zs,axis=1)
        scores_tot = evaluate([latent_fusion], labels_vector)
        print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format( scores_tot['kmeans']['accuracy'],
                                                                          scores_tot['kmeans']['NMI'],
                                                                          scores_tot['kmeans']['ARI'],
                                                                          scores_tot['kmeans']['pur']))

    return scores_each, scores_tot


