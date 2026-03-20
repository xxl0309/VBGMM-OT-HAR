import os
import glob
import numpy as np
import ot
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.spatial.distance import cdist
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


warnings.filterwarnings("ignore")

PAPER_PARAMS_GOLDEN = {
    'pca_dim': 50,
    'alpha': 0.01,
    'lambda1': 0.05,
    'class_w': 0.07,
    'total_clusters': 28,
    'smooth_win': 9,
    'balance_ratio': 1.0,
    'n_iter': 50,
    'conf_thresh': 0.75
}

CURRENT_PARAMS = PAPER_PARAMS_GOLDEN
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')


datasets = ['PAMAP2', 'DSADS', 'UCI', 'USC']


PAPER_NAMES = {
    'PAMAP2': 'PAMAP2',
    'DSADS': 'DSADS',
    'UCI': 'UCI',
    'USC': 'H'
}


def load_data(name):
    pattern = os.path.join(DATA_DIR, f"*_{name.lower()}_final.npy")
    files = glob.glob(pattern)
    if not files: return None, None
    d = np.load(files[0], allow_pickle=True).item()
    return d['X'], d['y']


def temporal_smoothing(preds, window_size=9):
    smoothed = np.copy(preds)
    pad = window_size // 2
    for i in range(len(preds)):
        start = max(0, i - pad)
        end = min(len(preds), i + pad + 1)
        window = preds[start:end]
        if len(window) > 0:
            smoothed[i] = np.bincount(window).argmax()
    return smoothed


class VerdictFGW:
    def __init__(self, params, seed=42):
        self.p = params
        self.seed = seed

    def fit_predict(self, X_s, y_s, X_t, y_t):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_s)
        X_t = scaler.fit_transform(X_t)
        pca = PCA(n_components=min(self.p['pca_dim'], X_s.shape[1]), whiten=False)
        X_s = pca.fit_transform(X_s)
        X_t = pca.transform(X_t)
        norm = Normalizer(norm='l2')
        X_s = norm.fit_transform(X_s)
        X_t = norm.transform(X_t)

        n_classes = 4
        c_per_class = self.p['total_clusters'] // n_classes
        means_s, weights_s, classes_s = [], [], []
        for cls in range(n_classes):
            X_cls = X_s[y_s == cls]
            gmm = BayesianGaussianMixture(
                n_components=c_per_class, covariance_type='diag', max_iter=500, n_init=10,
                random_state=self.seed, weight_concentration_prior=1e-3, reg_covar=0.01
            )
            gmm.fit(X_cls)
            means_s.append(gmm.means_)
            weights_s.append(gmm.weights_ * (len(X_cls) / len(X_s)))
            classes_s.extend([cls] * c_per_class)
        means_s = np.vstack(means_s)
        weights_s = np.hstack(weights_s) / np.sum(weights_s)
        classes_s = np.array(classes_s)

        kmeans = KMeans(n_clusters=self.p['total_clusters'], random_state=self.seed, n_init=30)
        kmeans.fit(X_t)
        means_t, labels_t = kmeans.cluster_centers_, kmeans.labels_
        weights_t = np.ones(self.p['total_clusters']) / self.p['total_clusters']

        C1 = cdist(means_s, means_s, metric='sqeuclidean')
        C1 /= C1.max()
        C2 = cdist(means_t, means_t, metric='sqeuclidean')
        C2 /= C2.max()
        M = cdist(means_s, means_t, metric='sqeuclidean')
        M /= M.max()

        Gs = ot.sinkhorn(weights_s, weights_t, M, self.p['lambda1'], numItermax=5000)
        y_pred_cluster = classes_s[np.argmax(Gs, axis=0)]

        for i in range(self.p['n_iter']):
            M_class = np.zeros_like(M)
            mask = (np.max(Gs, axis=0) > self.p['conf_thresh'])
            for k in range(self.p['total_clusters']):
                if mask[k]:
                    for row in range(self.p['total_clusters']):
                        if classes_s[row] != y_pred_cluster[k]:
                            M_class[row, k] = 1.0

            M_total = (1 - self.p['class_w']) * M + self.p['class_w'] * M_class
            if M_total.max() > 0: M_total /= M_total.max()

            Gs = ot.gromov.fused_gromov_wasserstein(
                M_total, C1, C2, weights_s, weights_t, 'square_loss', alpha=self.p['alpha'], verbose=False
            )
            y_pred_cluster = classes_s[np.argmax(Gs, axis=0)]

        y_pred = [classes_s[np.argmax(Gs[:, labels_t[j]])] for j in range(len(X_t))]
        return temporal_smoothing(np.array(y_pred), self.p['smooth_win'])



def save_confusion_matrix(y_true, y_pred, task_name):

    target_labels = ['lying', 'walking', 'ascending', 'descending']
    cm = confusion_matrix(y_true, y_pred, labels=[3, 0, 1, 2])

    # 归一化为比率
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7, 6))

    # 恢复为标准的无衬线字体，去掉了之前的粗体和 Times New Roman，对齐模板的清新风格
    plt.rcParams['font.family'] = 'sans-serif'

    # 绘制热力图
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                     xticklabels=target_labels, yticklabels=target_labels,
                     annot_kws={"size": 15}, vmin=0.0, vmax=1.0)

    # 严格对齐模板的文本：普通粗细，特定的大小写
    plt.title('Confusion matrix', fontsize=14, pad=15)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)

    # X和Y轴标签完全水平 (rotation=0)
    plt.xticks(fontsize=11, rotation=0)
    plt.yticks(fontsize=11, rotation=0)

    # 增加图表外边框 (对齐模板图周围的黑线)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

    plt.tight_layout()

    # 保存图片
    os.makedirs('CM_Images_Final_Format', exist_ok=True)
    safe_task_name = task_name.replace("->", "_to_")
    # 虽然标题里没有任务名了，但文件名里带有具体的任务名 (如 CM_PAMAP2_to_H.png)，方便你分辨
    plt.savefig(f'CM_Images_Final_Format/CM_{safe_task_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 65)
    print("正在运行 HAR 跨域识别 ")
    print("=" * 65)
    accs = []
    for source in datasets:
        for target in datasets:
            if source == target: continue
            X_s, y_s = load_data(source)
            X_t, y_t = load_data(target)
            if X_s is None or X_t is None: continue

            s_name = PAPER_NAMES[source]
            t_name = PAPER_NAMES[target]

            print(f" 处理 {s_name} -> {t_name}...", end=" ", flush=True)
            y_pred = VerdictFGW(CURRENT_PARAMS).fit_predict(X_s, y_s, X_t, y_t)
            acc = accuracy_score(y_t, y_pred) * 100
            accs.append(acc)
            print(f"准确率: {acc:.2f}%")

            # 传入任务名，但在画图时只作为文件名，不在图中显示
            save_confusion_matrix(y_t, y_pred, f"{s_name}->{t_name}")

    print("-" * 65)
    print(f"全局平均准确率: {np.mean(accs):.2f}%")


if __name__ == '__main__':
    main()