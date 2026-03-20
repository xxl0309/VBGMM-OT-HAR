import numpy as np
import pandas as pd
import ot
from scipy.spatial.distance import cdist
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings


warnings.filterwarnings("ignore")


from main_cm import load_data, temporal_smoothing


def balance_source_data(X, y, ratio=1.0):
    if ratio >= 1.0:
        return X, y
    classes = np.unique(y)
    min_samples = int(min([np.sum(y == c) for c in classes]) * ratio)
    if min_samples == 0: return X, y
    X_bal, y_bal = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) > min_samples:
            idx = np.random.choice(idx, min_samples, replace=False)
        X_bal.append(X[idx])
        y_bal.append(y[idx])
    return np.vstack(X_bal), np.hstack(y_bal)


PAPER_DEFAULT_PARAMS = {
    'pca_dim': 50,
    'lambda1': 0.05,
    'alpha': 0.01,
    'class_w': 0.07,
    'total_clusters': 28,
    'n_iter': 50,
    'smooth_win': 9,
    'conf_thresh': 0.75,
    'balance_ratio': 1.0
}

datasets = ['PAMAP2', 'DSADS', 'UCI', 'USC']


class Evaluator:
    def __init__(self, params, seed=42):
        self.p = params
        self.seed = seed

    def _fit_source(self, X, y):
        n_classes = 4
        clusters_per_class = self.p['total_clusters'] // n_classes
        means, weights, classes = [], [], []

        for cls in range(n_classes):
            X_cls = X[y == cls]
            gmm = BayesianGaussianMixture(
                n_components=clusters_per_class, covariance_type='diag',
                weight_concentration_prior=1e-3, max_iter=300, n_init=3,
                init_params='kmeans', random_state=self.seed, reg_covar=0.01
            )
            gmm.fit(X_cls)
            means.append(gmm.means_)
            weights.append(gmm.weights_ * (len(X_cls) / len(X)))
            classes.extend([cls] * clusters_per_class)
        return np.vstack(means), np.hstack(weights) / np.sum(weights), np.array(classes)

    def run_one_pair(self, s, t):
        try:
            X_s, y_s = load_data(s)
            X_t, y_t = load_data(t)
            if X_s is None or X_t is None: return None

            # 1. Preprocessing
            X_s, y_s = balance_source_data(X_s, y_s, ratio=self.p['balance_ratio'])
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_s)
            X_t = scaler.fit_transform(X_t)
            pca = PCA(n_components=min(self.p['pca_dim'], X_s.shape[1]), whiten=False)
            X_s = pca.fit_transform(X_s)
            X_t = pca.transform(X_t)
            normalizer = Normalizer(norm='l2')
            X_s = normalizer.fit_transform(X_s)
            X_t = normalizer.transform(X_t)

            # 2. Source & Target Clusters
            means_s, weights_s, classes_s = self._fit_source(X_s, y_s)
            kmeans = KMeans(n_clusters=self.p['total_clusters'], random_state=self.seed, n_init=10)
            kmeans.fit(X_t)
            means_t, labels_t = kmeans.cluster_centers_, kmeans.labels_
            weights_t = np.ones(self.p['total_clusters']) / self.p['total_clusters']

            # 3. Cost Matrices
            C1 = cdist(means_s, means_s, metric='sqeuclidean')
            C1 /= C1.max()
            C2 = cdist(means_t, means_t, metric='sqeuclidean')
            C2 /= C2.max()
            M = cdist(means_s, means_t, metric='sqeuclidean')
            M /= M.max()

            Gs = ot.sinkhorn(weights_s, weights_t, M, self.p['lambda1'], numItermax=5000)
            y_pred_cluster = classes_s[np.argmax(Gs, axis=0)]

            # 4. FGW Loop
            for i in range(self.p['n_iter']):
                M_class = np.zeros_like(M)
                mask = (np.max(Gs, axis=0) > self.p['conf_thresh'])

                for k in range(self.p['total_clusters']):
                    if mask[k]:
                        for row in range(self.p['total_clusters']):
                            if classes_s[row] != y_pred_cluster[k]:
                                M_class[row, k] = 1.0

                cw = self.p['class_w']
                if cw < 1.0:
                    M_total = (1 - cw) * M + cw * M_class
                else:
                    M_total = M + cw * M_class

                if M_total.max() > 0: M_total /= M_total.max()

                Gs = ot.gromov.fused_gromov_wasserstein(
                    M_total, C1, C2, weights_s, weights_t, 'square_loss', alpha=self.p['alpha'], verbose=False
                )
                y_pred_cluster = classes_s[np.argmax(Gs, axis=0)]

            y_pred = [classes_s[np.argmax(Gs[:, labels_t[j]])] for j in range(len(X_t))]
            return accuracy_score(y_t, temporal_smoothing(np.array(y_pred), self.p['smooth_win'])) * 100
        except Exception as e:
            return None

    def run_global_avg(self):
        accs = []
        for s in datasets:
            for t in datasets:
                if s == t: continue
                res = self.run_one_pair(s, t)
                if res is not None: accs.append(res)
        return np.mean(accs)

# ========================================================

def run_full_sweep():
    print("正在运行参数敏感性分析 (Parameter Sensitivity Analysis)...")

    results = []

    # 1. 测维度 d
    print("[1/4] Testing PCA Dimension (d)...")
    for d in [30, 40, 50, 60]:
        p = PAPER_DEFAULT_PARAMS.copy()
        p['pca_dim'] = d
        acc = Evaluator(p).run_global_avg()
        results.append({'Type': 'Dimension', 'Value': d, 'Accuracy': acc})
        print(f"  d={d}: {acc:.2f}%")

    # 2. 测结构权重 alpha/eta
    print("\n[2/4] Testing Structure Weight (alpha/eta)...")
    for a in [0.005, 0.01, 0.05, 0.1, 0.2]:
        p = PAPER_DEFAULT_PARAMS.copy()
        p['alpha'] = a
        acc = Evaluator(p).run_global_avg()
        results.append({'Type': 'Structure', 'Value': a, 'Accuracy': acc})
        print(f"  alpha={a}: {acc:.2f}%")

    # 3. 测熵正则 lambda
    print("\n[3/4] Testing Entropy Reg (lambda)...")
    for l in [0.01, 0.02, 0.05, 0.1]:
        p = PAPER_DEFAULT_PARAMS.copy()
        p['lambda1'] = l
        acc = Evaluator(p).run_global_avg()
        results.append({'Type': 'Entropy', 'Value': l, 'Accuracy': acc})
        print(f"  lambda={l}: {acc:.2f}%")

    # 4. 测类别权重 class_w
    print("\n[4/4] Testing Class Weight (lambda_1)...")
    for w in [0.01, 0.05, 0.07, 0.1, 0.5]:
        p = PAPER_DEFAULT_PARAMS.copy()
        p['class_w'] = w
        acc = Evaluator(p).run_global_avg()
        results.append({'Type': 'ClassW', 'Value': w, 'Accuracy': acc})
        print(f"  class_w={w}: {acc:.2f}%")

    pd.DataFrame(results).to_csv("paper_params_sensitivity.csv", index=False)
    print("\n测试完毕！结果已保存到 paper_params_sensitivity.csv")

if __name__ == '__main__':
    run_full_sweep()