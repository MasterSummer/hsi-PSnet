import os
import random
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ============== 设置随机种子 ==============
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============== 加载数据 (与你原脚本相同) ==============
trainval_pairs = torch.load("splits/trainval.pt")
test_pairs = torch.load("splits/cleantest.pt")

X_trainval = torch.stack([p[1] for p in trainval_pairs])  # HSI 数据
y_trainval = torch.tensor([p[2] for p in trainval_pairs])

X_test = torch.stack([p[1] for p in test_pairs])
y_test = torch.tensor([p[2] for p in test_pairs])

# 展平为 (n_samples, n_features)
X_trainval_np = X_trainval.permute(0, 2, 3, 1).reshape(
    -1, X_trainval.shape[1] * X_trainval.shape[2] * X_trainval.shape[3]
).numpy()
y_trainval_np = y_trainval.numpy()

X_test_np = X_test.permute(0, 2, 3, 1).reshape(
    -1, X_test.shape[1] * X_test.shape[2] * X_test.shape[3]
).numpy()
y_test_np = y_test.numpy()

# ============== 辅助函数：计算指标 ==============
def compute_metrics(y_true, y_pred, y_prob=None):
    """返回 dict: Accuracy, Precision, Recall, F1, AUROC（可能为 nan）"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auroc = np.nan
    if y_prob is not None:
        try:
            auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auroc = np.nan
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1, "AUROC": auroc}

# ============== 主函数：跑实验 ==============
def run_experiment(model_name, model_init, X_trainval, y_trainval, X_test, y_test, pca_dim=30, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # 存放每折的验证集指标（cv）与每折在外部测试集上的指标（test_via_cv）
    cv_val_metrics = {k: [] for k in ["Accuracy", "Precision", "Recall", "F1-score", "AUROC"]}
    cv_test_metrics = {k: [] for k in ["Accuracy", "Precision", "Recall", "F1-score", "AUROC"]}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
        print(f"[{model_name}] Fold {fold+1}/{n_splits} training...")
        X_tr, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_tr, y_val = y_trainval[train_idx], y_trainval[val_idx]

        # 标准化（用训练折拟合）
        scaler = StandardScaler()
        X_tr_std = scaler.fit_transform(X_tr)
        X_val_std = scaler.transform(X_val)
        X_test_std_for_fold = scaler.transform(X_test)  # 用当前折的 scaler -> 用来评估外部测试集

        # PCA（用训练折拟合）
        pca = PCA(n_components=pca_dim, random_state=SEED)
        X_tr_pca = pca.fit_transform(X_tr_std)
        X_val_pca = pca.transform(X_val_std)
        X_test_pca_for_fold = pca.transform(X_test_std_for_fold)  # 用当前折的 pca -> 用来评估外部测试集

        # 训练模型（当前折）
        clf = model_init()
        clf.fit(X_tr_pca, y_tr)

        # 在验证集上的评估（fold 内）
        y_val_pred = clf.predict(X_val_pca)
        # 得到概率或用 decision_function -> 转成概率近似
        if hasattr(clf, "predict_proba"):
            y_val_prob = clf.predict_proba(X_val_pca)
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_val_pca)
            # 二分类特殊处理，确保 shape=(n_samples, n_classes)
            if scores.ndim == 1:
                probs_pos = 1 / (1 + np.exp(-scores))
                y_val_prob = np.vstack([1 - probs_pos, probs_pos]).T
            else:
                exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                y_val_prob = exps / np.sum(exps, axis=1, keepdims=True)
        else:
            y_val_prob = None

        val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
        for k in cv_val_metrics:
            cv_val_metrics[k].append(val_metrics[k])

        # 用该折模型在外部测试集上的评估 -> 这是你要的“测试集方差”来源
        y_test_pred_fold = clf.predict(X_test_pca_for_fold)
        if hasattr(clf, "predict_proba"):
            y_test_prob_fold = clf.predict_proba(X_test_pca_for_fold)
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_test_pca_for_fold)
            if scores.ndim == 1:
                probs_pos = 1 / (1 + np.exp(-scores))
                y_test_prob_fold = np.vstack([1 - probs_pos, probs_pos]).T
            else:
                exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                y_test_prob_fold = exps / np.sum(exps, axis=1, keepdims=True)
        else:
            y_test_prob_fold = None

        test_metrics_fold = compute_metrics(y_test, y_test_pred_fold, y_test_prob_fold)
        for k in cv_test_metrics:
            cv_test_metrics[k].append(test_metrics_fold[k])

        print(f"  fold {fold+1} val acc: {val_metrics['Accuracy']:.4f}, test(acc by this fold): {test_metrics_fold['Accuracy']:.4f}")

    # 交叉验证（验证集）mean ± std
    cv_val_mean = {k: np.nanmean(v) for k, v in cv_val_metrics.items()}
    cv_val_std  = {k: np.nanstd(v)  for k, v in cv_val_metrics.items()}

    # CV-trained models 在外部测试集上的 mean ± std（你要的“测试集方差”）
    cv_test_mean = {k: np.nanmean(v) for k, v in cv_test_metrics.items()}
    cv_test_std  = {k: np.nanstd(v)  for k, v in cv_test_metrics.items()}

    # ========== 用所有训练数据训练最终模型并评估外部测试集 ==========
    # 标准化 + PCA 基于全部 trainval 拟合
    scaler_all = StandardScaler()
    X_trainval_std_all = scaler_all.fit_transform(X_trainval)
    X_test_std_all = scaler_all.transform(X_test)

    pca_all = PCA(n_components=pca_dim, random_state=SEED)
    X_trainval_pca_all = pca_all.fit_transform(X_trainval_std_all)
    X_test_pca_all = pca_all.transform(X_test_std_all)

    clf_final = model_init()
    clf_final.fit(X_trainval_pca_all, y_trainval)

    y_test_pred_final = clf_final.predict(X_test_pca_all)
    if hasattr(clf_final, "predict_proba"):
        y_test_prob_final = clf_final.predict_proba(X_test_pca_all)
    elif hasattr(clf_final, "decision_function"):
        scores = clf_final.decision_function(X_test_pca_all)
        if scores.ndim == 1:
            probs_pos = 1 / (1 + np.exp(-scores))
            y_test_prob_final = np.vstack([1 - probs_pos, probs_pos]).T
        else:
            exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            y_test_prob_final = exps / np.sum(exps, axis=1, keepdims=True)
    else:
        y_test_prob_final = None

    final_test_metrics = compute_metrics(y_test, y_test_pred_final, y_test_prob_final)

    # ========== 保存结果到文件 ==========
    os.makedirs(f"result/{model_name}", exist_ok=True)
    save_path = os.path.join("result", model_name, "results.txt")
    with open(save_path, "w") as f:
        f.write("===== Cross-validation (validation folds) =====\n")
        for k in cv_val_mean.keys():
            f.write(f"{k}: {cv_val_mean[k]:.4f} ± {cv_val_std[k]:.4f}\n")

        f.write("\n===== CV-trained models evaluated on external TEST set (mean ± std) =====\n")
        for k in cv_test_mean.keys():
            f.write(f"{k}: {cv_test_mean[k]:.4f} ± {cv_test_std[k]:.4f}\n")

        f.write("\n===== Final model (trained on all trainval) evaluated on TEST set =====\n")
        for k, v in final_test_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"✅ {model_name} results saved to: {save_path}")
    return {
        "cv_val_mean": cv_val_mean, "cv_val_std": cv_val_std,
        "cv_test_mean": cv_test_mean, "cv_test_std": cv_test_std,
        "final_test": final_test_metrics
    }

# ============== 运行 RF 和 SVM ==============
results_rf = run_experiment(
    model_name="RF",
    model_init=lambda: RandomForestClassifier(n_estimators=100, random_state=SEED),
    X_trainval=X_trainval_np,
    y_trainval=y_trainval_np,
    X_test=X_test_np,
    y_test=y_test_np,
    pca_dim=30,
    n_splits=5
)

results_svm = run_experiment(
    model_name="SVM",
    model_init=lambda: SVC(C=1.0, kernel="rbf", probability=True, random_state=SEED),
    X_trainval=X_trainval_np,
    y_trainval=y_trainval_np,
    X_test=X_test_np,
    y_test=y_test_np,
    pca_dim=30,
    n_splits=5
)
