import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import joblib

from joblib import dump
from imblearn.over_sampling import SMOTEN

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


# =========================================================
# 设置中文字体
# =========================================================
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# =========================================================
# 定义标准化处理函数
# =========================================================
def apply_standardization(df):
    """
    对输入特征进行标准化处理。

    参数：
        df: pandas DataFrame，原始特征数据

    返回：
        df_standardized: pandas DataFrame，标准化后的特征数据
        scaler: StandardScaler，训练好的标准化器
    """
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    return df_standardized, scaler


# =========================================================
# 路径设置：使用相对路径，便于上传 Zenodo 后复现
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

input_base_path = BASE_DIR / 'data'
output_base_path_model = BASE_DIR / 'output' / 'model'
output_base_path_metrics = BASE_DIR / 'output' / 'metrics'
output_base_path_metrics_all = BASE_DIR / 'output' / 'metrics_all'
output_base_path_metrics_all_com = BASE_DIR / 'output' / 'metrics_all_com'

for folder in [
    input_base_path,
    output_base_path_model,
    output_base_path_metrics,
    output_base_path_metrics_all,
    output_base_path_metrics_all_com,
]:
    folder.mkdir(parents=True, exist_ok=True)

input_file = input_base_path / 'CS15_input_right_data.xlsx'
input_file_y_new_true = input_base_path / 'CS15_input_right_new_sample_labels.xlsx'
input_file_X_new_samples = input_base_path / 'CS15_input_right_new_sample.xlsx'


# =========================================================
# 读取数据
# =========================================================
data = pd.read_excel(input_file)

# 将目标变量转换为分类标签
# 原始变形宽度 < -6 m 记为崩退样本 1，否则记为非崩退样本 0
data.iloc[:, 1] = (data.iloc[:, 1] < -6).astype(int)

# 提取样本信息列
sample_info = data.iloc[:, -1]
features_data = data.iloc[:, 2:-1]


# =========================================================
# 自相关检验：删除相关系数大于 0.9 的特征
# =========================================================
correlation_matrix = features_data.corr().abs()

upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

high_corr_pairs = []
to_drop = set()

for col in upper_triangle.columns:
    high_corr = upper_triangle[col][upper_triangle[col] > 0.9].index.tolist()
    for feature in high_corr:
        corr_value = correlation_matrix.loc[col, feature]
        high_corr_pairs.append((col, feature, corr_value))
        to_drop.add(feature)

print("\n高相关性特征对（相关系数 > 0.9）：")
for pair in high_corr_pairs:
    print(f"特征 '{pair[0]}' 和 '{pair[1]}' 相关系数: {pair[2]:.4f} → 删除 '{pair[1]}'")

to_drop = list(to_drop)
filtered_features_data = features_data.drop(columns=to_drop)

print("\n最终删除的特征：", to_drop)

# 更新数据集，仅保留筛选后的特征，并保留样本信息列
data = pd.concat(
    [data.iloc[:, :2], filtered_features_data, sample_info],
    axis=1
)

# 按年份排序
data = data.sort_values(by=data.columns[0]).reset_index(drop=True)


# =========================================================
# 划分训练验证集和测试集
# =========================================================
X_train_val_raw = data.iloc[:-10, 2:-1]
y_train_val = data.iloc[:-10, 1].reset_index(drop=True)

X_test_raw = data.iloc[-10:, 2:-1]
y_test = data.iloc[-10:, 1].reset_index(drop=True)

test_sample_info = data.iloc[-10:, -1].reset_index(drop=True)


# =========================================================
# 标准化处理
# =========================================================
X_train_val_standardized, scaler_x = apply_standardization(X_train_val_raw)

X_test_standardized = pd.DataFrame(
    scaler_x.transform(X_test_raw),
    columns=X_train_val_raw.columns,
    index=X_test_raw.index
).reset_index(drop=True)

X_train_val_standardized = X_train_val_standardized.reset_index(drop=True)


# =========================================================
# 读取 New_sample
# =========================================================
new_samples_data = pd.read_excel(input_file_X_new_samples)
filtered_new_samples_data = new_samples_data.drop(columns=to_drop)

# 从第三列开始提取特征
# 第二列为空“变形宽度”，各列保持与训练数据一致
X_new_raw = filtered_new_samples_data.iloc[:, 2:]

X_new = pd.DataFrame(
    scaler_x.transform(X_new_raw),
    columns=X_train_val_raw.columns
)

# 读取 New_sample 的真实值
y_new_true = pd.read_excel(input_file_y_new_true)['真实值']
y_new_true = (y_new_true < -6).astype(int)


# =========================================================
# 将后续建模数据统一命名为标准化后的数据
# =========================================================
X_train_val = X_train_val_standardized
X_test = X_test_standardized


# =========================================================
# 检查数据维度
# =========================================================
print(f"\nX_train_val shape: {X_train_val.shape}")
print(f"y_train_val shape: {y_train_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"X_new shape: {X_new.shape}")
print(f"y_new_true shape: {y_new_true.shape}")

n_train = int(0.75 * len(X_train_val))


# =========================================================
# 检查训练集中目标变量分布，并根据需要进行 SMOTEN 重采样
# 注意：
# 这里对标准化后的 X_train_val 进行重采样
# =========================================================
positive_samples = np.sum(y_train_val == 1)
negative_samples = np.sum(y_train_val == 0)

pos_neg_ratio = positive_samples / negative_samples if negative_samples != 0 else float('inf')

print(f"\n训练集中正样本数: {positive_samples}, 负样本数: {negative_samples}, 比例: {pos_neg_ratio:.2f}")

imbalance_threshold = 0.8

if pos_neg_ratio <= imbalance_threshold or pos_neg_ratio >= 1 / imbalance_threshold:
    print("检测到样本不平衡，应用 SMOTEN 进行重采样...")

    smoten = SMOTEN(random_state=42)
    X_resampled, y_resampled = smoten.fit_resample(X_train_val, y_train_val)

    X_train_val = pd.DataFrame(
        X_resampled,
        columns=X_train_val.columns
    )

    y_train_val = pd.Series(
        y_resampled,
        name='target_class'
    )

    print(f"重采样后训练集样本数: {len(y_train_val)}")

else:
    print("样本较为平衡，无需重采样。")


# =========================================================
# 学习曲线函数
# =========================================================
def plot_learning_curve(
    estimator,
    n_features,
    X_train_val_selected,
    X_test_selected,
    y_train_val,
    y_test,
    param_name,
    param_range
):
    plt.clf()

    train_scores = []
    val_scores = []
    val_recalls = []

    test_scores = []
    test_recalls = []

    cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=42
    )

    for param in param_range:
        estimator.set_params(n_neighbors=param)

        fold_train_scores = []
        fold_val_scores = []
        fold_val_recalls = []

        for train_idx, val_idx in cv.split(X_train_val_selected, y_train_val):
            X_train_cv = X_train_val_selected.iloc[train_idx]
            X_val_cv = X_train_val_selected.iloc[val_idx]

            y_train_cv = y_train_val.iloc[train_idx]
            y_val_cv = y_train_val.iloc[val_idx]

            estimator.fit(X_train_cv, y_train_cv)

            y_train_cv_pred = estimator.predict(X_train_cv)
            y_val_cv_pred = estimator.predict(X_val_cv)

            fold_train_scores.append(
                accuracy_score(y_train_cv, y_train_cv_pred)
            )
            fold_val_scores.append(
                accuracy_score(y_val_cv, y_val_cv_pred)
            )
            fold_val_recalls.append(
                recall_score(y_val_cv, y_val_cv_pred)
            )

        train_scores.append(np.mean(fold_train_scores))
        val_scores.append(np.mean(fold_val_scores))
        val_recalls.append(np.mean(fold_val_recalls))

        # 测试集评估
        estimator.fit(X_train_val_selected, y_train_val)
        y_test_pred = estimator.predict(X_test_selected)

        test_scores.append(
            accuracy_score(y_test, y_test_pred)
        )
        test_recalls.append(
            recall_score(y_test, y_test_pred)
        )

    # 绘制包含 Accuracy 和 Recall 的学习曲线
    plt.plot(
        param_range,
        train_scores,
        label='Training Accuracy',
        marker='o',
        color='blue'
    )

    plt.plot(
        param_range,
        val_scores,
        label='Validation Accuracy',
        marker='x',
        color='purple'
    )

    plt.plot(
        param_range,
        test_scores,
        label='Test Accuracy',
        marker='D',
        color='red'
    )

    plt.plot(
        param_range,
        val_recalls,
        label='Validation Recall',
        marker='s',
        color='green'
    )

    plt.plot(
        param_range,
        test_recalls,
        label='Test Recall',
        marker='p',
        color='brown'
    )

    plt.title(f'Learning Curve (n_features: {n_features}, {param_name})')
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_fig_file_all = output_base_path_metrics_all / f'n_features_{n_features}.png'
    plt.savefig(output_fig_file_all)

    # 绘制只包含 Accuracy 的学习曲线
    plt.figure()

    plt.plot(
        param_range,
        train_scores,
        label='Training Accuracy',
        marker='o',
        color='blue'
    )

    plt.plot(
        param_range,
        val_scores,
        label='Validation Accuracy',
        marker='x',
        color='purple'
    )

    plt.plot(
        param_range,
        test_scores,
        label='Test Accuracy',
        marker='D',
        color='red'
    )

    plt.title(f'Learning Curve (n_features: {n_features}, {param_name})')
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_fig_file_a = output_base_path_metrics / f'n_features_{n_features}.png'
    plt.savefig(output_fig_file_a)

    plt.close('all')

    return val_scores, val_recalls, test_scores, test_recalls


# =========================================================
# 依次测试不同特征数量下的学习曲线
# =========================================================
combined_results = pd.DataFrame({
    'n_neighbors': list(range(1, n_train))
})

# 保存最后一次 Lasso 排序结果
final_sorted_indices = None

for selected_feature_count in range(1, 28):

    print("\nnumber of the selected feature names:", selected_feature_count)

    X_input = X_train_val
    y_input = y_train_val

    # Lasso 在标准化后的训练数据上进行特征排序
    lasso = Lasso(
        alpha=0.01,
        random_state=42,
        max_iter=5000
    )

    lasso.fit(X_input, y_input)

    feature_importances = np.abs(lasso.coef_)
    sorted_indices = np.argsort(feature_importances)[::-1]
    final_sorted_indices = sorted_indices.copy()

    top_indices = sorted_indices[:selected_feature_count]

    X_train_val_selected = X_train_val.iloc[:, top_indices]
    X_test_selected = X_test.iloc[:, top_indices]

    selected_feature_names = X_train_val.columns[top_indices]

    print("当前选择的特征：")
    print(list(selected_feature_names))

    knn = KNeighborsClassifier(
        metric='manhattan',
        weights='distance'
    )

    param_range = range(1, n_train)

    val_scores, val_recalls, test_scores, test_recalls = plot_learning_curve(
        estimator=knn,
        n_features=selected_feature_count,
        X_train_val_selected=X_train_val_selected,
        X_test_selected=X_test_selected,
        y_train_val=y_train_val,
        y_test=y_test,
        param_name='n_neighbors',
        param_range=param_range
    )

    new_columns = {
        f'Val_Acc_{selected_feature_count}': val_scores,
        f'Val_Recall_{selected_feature_count}': val_recalls,
        f'Test_Acc_{selected_feature_count}': test_scores,
        f'Test_Recall_{selected_feature_count}': test_recalls
    }

    new_df = pd.DataFrame(new_columns)
    combined_results = pd.concat([combined_results, new_df], axis=1)


# 保存不同特征数量和不同 K 值下的结果
combined_results_file = output_base_path_metrics_all_com / 'combined_learning_curve_results.xlsx'
combined_results.to_excel(combined_results_file, index=False)

print(f"\n学习曲线结果已保存至: {combined_results_file}")


# =========================================================
# 确定最终特征数量和 K 值
# =========================================================
final_n_features = 11
n_neighbors = 2

if final_sorted_indices is None:
    raise ValueError("Lasso 特征排序结果为空，请检查输入数据。")

best_selected_features = final_sorted_indices[:final_n_features]

print("\n最终选择的特征索引：")
print(best_selected_features)

print("\n最终选择的特征名称：")
print(list(X_train_val.columns[best_selected_features]))

X_train_val_final = X_train_val.iloc[:, best_selected_features]
X_test_final = X_test.iloc[:, best_selected_features]


# =========================================================
# 初始化最终 KNN 分类模型
# =========================================================
knn = KNeighborsClassifier(
    n_neighbors=n_neighbors,
    metric='manhattan',
    weights='distance'
)


# =========================================================
# 自定义函数计算 Specificity
# =========================================================
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


# =========================================================
# 4 折交叉验证
# =========================================================
kf = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

train_acc_scores = []
train_precision_scores = []
train_recall_scores = []
train_f1_scores = []
train_specificity_scores = []

val_acc_scores = []
val_precision_scores = []
val_recall_scores = []
val_f1_scores = []
val_specificity_scores = []

for fold, (train_idx, val_idx) in enumerate(
    kf.split(X_train_val_final, y_train_val)
):
    X_train_final = X_train_val_final.iloc[train_idx]
    X_val_final = X_train_val_final.iloc[val_idx]

    y_train = y_train_val.iloc[train_idx]
    y_val = y_train_val.iloc[val_idx]

    # 训练模型
    knn.fit(X_train_final, y_train)

    # 训练集评估
    y_train_pred = knn.predict(X_train_final)

    train_acc_scores.append(
        accuracy_score(y_train, y_train_pred)
    )
    train_precision_scores.append(
        precision_score(y_train, y_train_pred, zero_division=0)
    )
    train_recall_scores.append(
        recall_score(y_train, y_train_pred, zero_division=0)
    )
    train_f1_scores.append(
        f1_score(y_train, y_train_pred, zero_division=0)
    )
    train_specificity_scores.append(
        specificity_score(y_train, y_train_pred)
    )

    # 验证集评估
    y_val_pred = knn.predict(X_val_final)

    val_acc_scores.append(
        accuracy_score(y_val, y_val_pred)
    )
    val_precision_scores.append(
        precision_score(y_val, y_val_pred, zero_division=0)
    )
    val_recall_scores.append(
        recall_score(y_val, y_val_pred, zero_division=0)
    )
    val_f1_scores.append(
        f1_score(y_val, y_val_pred, zero_division=0)
    )
    val_specificity_scores.append(
        specificity_score(y_val, y_val_pred)
    )

    print(f"\nFold {fold + 1} 结果:")
    print(
        f"Train Accuracy: {train_acc_scores[-1]:.4f}, "
        f"Specificity: {train_specificity_scores[-1]:.4f}, "
        f"Recall: {train_recall_scores[-1]:.4f}, "
        f"F1: {train_f1_scores[-1]:.4f}"
    )
    print(
        f"Val   Accuracy: {val_acc_scores[-1]:.4f}, "
        f"Specificity: {val_specificity_scores[-1]:.4f}, "
        f"Recall: {val_recall_scores[-1]:.4f}, "
        f"F1: {val_f1_scores[-1]:.4f}"
    )
    print("-" * 40)


# =========================================================
# 输出交叉验证平均结果
# =========================================================
print("\n交叉验证平均结果（Cross-Validation Average Results）:")

print(f"训练集 Accuracy: {np.mean(train_acc_scores):.4f}")
print(f"训练集 Specificity: {np.mean(train_specificity_scores):.4f}")
print(f"训练集 Recall: {np.mean(train_recall_scores):.4f}")
print(f"训练集 F1-score: {np.mean(train_f1_scores):.4f}")

print(f"验证集 Accuracy: {np.mean(val_acc_scores):.4f}")
print(f"验证集 Specificity: {np.mean(val_specificity_scores):.4f}")
print(f"验证集 Recall: {np.mean(val_recall_scores):.4f}")
print(f"验证集 F1-score: {np.mean(val_f1_scores):.4f}")


# =========================================================
# 在整个训练验证集上训练最终模型
# =========================================================
knn.fit(X_train_val_final, y_train_val)


# =========================================================
# 测试集评估
# =========================================================
y_test_pred = knn.predict(X_test_final)

test_acc = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_specificity = specificity_score(y_test, y_test_pred)

print("\n测试集模型表现（Final Test Performance）:")
print(f"测试集 Accuracy: {test_acc:.4f}")
print(f"测试集 Specificity: {test_specificity:.4f}")
print(f"测试集 Recall: {test_recall:.4f}")
print(f"测试集 F1-score: {test_f1:.4f}")


# =========================================================
# 输出测试集真实值与预测值
# =========================================================
print("\n测试集真实值与预测值：")

comparison = pd.DataFrame({
    '样本信息': test_sample_info,
    '真实值': y_test,
    '预测值': y_test_pred
})

print(comparison)

comparison_file = output_base_path_metrics / 'test_prediction_comparison.xlsx'
comparison.to_excel(comparison_file, index=False)

print(f"测试集预测结果已保存至: {comparison_file}")


# =========================================================
# 打印测试集混淆矩阵
# =========================================================
print("\n===== 测试集混淆矩阵 =====")
print(confusion_matrix(y_test, y_test_pred, labels=[0, 1]))


# =========================================================
# 保存模型、标准化器和最终特征索引
# =========================================================
model_path = output_base_path_model / 'left_model_KNN_c_202505.pkl'
dump(knn, model_path)
print(f"\n模型已保存至: {model_path}")

scalerX_path = output_base_path_model / 'left_scalerX_KNN_c_202505.pkl'
joblib.dump(scaler_x, scalerX_path)
print(f"标准化器已保存至: {scalerX_path}")

selected_features_path = output_base_path_model / 'selected_feature_indices.pkl'
joblib.dump(best_selected_features, selected_features_path)
print(f"最终特征索引已保存至: {selected_features_path}")


# =========================================================
# 加载模型、标准化器和最终特征索引
# =========================================================
best_model = joblib.load(model_path)
scaler_x = joblib.load(scalerX_path)
best_selected_features = joblib.load(selected_features_path)


# =========================================================
# New_sample 预测
# 已经使用训练集 scaler_x 标准化
# =========================================================
def predict_new_samples_classification(X_new, y_new_true):
    all_feature_names = X_new.columns.tolist()

    # 选择训练集最终确定的最佳特征
    X_new_selected = X_new.iloc[:, best_selected_features]

    # 输出预测时使用的特征名称
    X_new_feature_names = [
        all_feature_names[i] for i in best_selected_features
    ]

    print("\n预测 New_sample 时使用的特征名称：")
    print(', '.join(X_new_feature_names))

    # 预测类别
    y_pred = best_model.predict(X_new_selected)

    # 预测概率：第二列是类别 1 的概率，即崩退概率
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_new_selected)[:, 1]
    else:
        y_proba = None

    print()
    print(
        "New_sample 真实值:",
        y_new_true.values if hasattr(y_new_true, 'values') else y_new_true
    )
    print("New_sample 预测值:", y_pred)

    if y_proba is not None:
        print("预测为崩退的概率 P(y=1):", y_proba.round(4))
    else:
        print("该模型不支持概率输出 predict_proba。")

    # 分类指标评估
    acc = accuracy_score(y_new_true, y_pred)

    print("\nNew_sample 分类指标评估：")
    print(f"准确率 Accuracy: {acc:.4f}")

    # 保存 New_sample 预测结果
    new_sample_results = pd.DataFrame({
        '真实值': y_new_true,
        '预测值': y_pred
    })

    if y_proba is not None:
        new_sample_results['崩退概率_P_y_1'] = y_proba

    output_new_sample_file = output_base_path_metrics / 'new_sample_prediction_results.xlsx'
    new_sample_results.to_excel(output_new_sample_file, index=False)

    print(f"New_sample 预测结果已保存至: {output_new_sample_file}")


predict_new_samples_classification(X_new, y_new_true)