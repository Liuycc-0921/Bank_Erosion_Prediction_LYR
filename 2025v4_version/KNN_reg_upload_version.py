# ======================== 导入库 ========================
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import joblib

from joblib import dump
from scipy.stats import skew, kurtosis
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, KernelDensity
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ======================== 工具函数与预处理 ========================

def apply_standardization(df):
    """
    对 X 特征进行标准化处理。
    注意：
    1. scaler 只在训练验证集上 fit；
    2. 测试集和 New_sample 只能使用同一个 scaler transform；
    3. 后续 Lasso、采样、KNN 训练和预测全部使用标准化后的 X。
    """
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    return df_standardized, scaler


def fit_y_transformer(y):
    """
    基于当前 y 拟合 Yeo-Johnson 变换器。
    这里 y 可以包含负值，所以使用 Yeo-Johnson。
    """
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    pt.fit(np.asarray(y).reshape(-1, 1))
    return pt


def y_transform(pt, y):
    """
    将 y 从原始空间变换到近似正态空间。
    """
    return pt.transform(np.asarray(y).reshape(-1, 1)).ravel()


def y_inverse(pt, y_transformed):
    """
    将 y 从变换空间逆变换回原始空间。
    """
    return pt.inverse_transform(np.asarray(y_transformed).reshape(-1, 1)).ravel()


# ======================== 路径与文件：改为相对路径 ========================

BASE_DIR = Path(__file__).resolve().parent

input_base_path = BASE_DIR / 'data'

output_base_path_model = BASE_DIR / 'output' / 'model'
output_base_path_weight = BASE_DIR / 'output' / 'weight'
output_base_path_curves_mae = BASE_DIR / 'output' / 'curves_mae'
output_base_path_curves_smape = BASE_DIR / 'output' / 'curves_smape'
output_base_path_curves_mae_adjusted = BASE_DIR / 'output' / 'curves_mae_adjusted'
output_base_path_curves_smape_adjusted = BASE_DIR / 'output' / 'curves_smape_adjusted'
output_base_path_metrics = BASE_DIR / 'output' / 'metrics'

for folder in [
    input_base_path,
    output_base_path_model,
    output_base_path_weight,
    output_base_path_curves_mae,
    output_base_path_curves_smape,
    output_base_path_curves_mae_adjusted,
    output_base_path_curves_smape_adjusted,
    output_base_path_metrics,
]:
    folder.mkdir(parents=True, exist_ok=True)

input_file = input_base_path / '3_CS15_input_right_data_case3_de11.xlsx'
input_file_y_new_true = input_base_path / 'CS15_input_right_new_sample_labels.xlsx'
input_file_X_new_samples = input_base_path / 'CS15_input_right_new_sample.xlsx'
input_file_X_new_samples_work_constraint = input_base_path / 'CS15_input_right_new_sample_work_distance.xlsx'


# ======================== 数据读取与特征筛选 ========================

data = pd.read_excel(input_file)

# 筛选目标变量小于 -6 的样本
# 回归模型只对已经发生崩退的样本进行崩退宽度预测
data = data[data.iloc[:, 1] < -6].copy()

# 提取样本信息列
sample_info = data.iloc[:, -1]
features_data = data.iloc[:, 2:-1]

# 自相关检验：删除相关系数大于 0.9 的特征
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

# 按年份排序，并重置索引
data = data.sort_values(by=data.columns[0]).reset_index(drop=True)


# ======================== 划分训练/验证集和测试集 ========================

X_train_val_raw = data.iloc[:-3, 2:-1]
y_train_val = data.iloc[:-3, 1].reset_index(drop=True)

X_test_raw = data.iloc[-3:, 2:-1]
y_test = data.iloc[-3:, 1].reset_index(drop=True)

test_sample_info = data.iloc[-3:, -1].reset_index(drop=True)


# ======================== X 标准化：确保训练、测试、New_sample 尺度一致 ========================

X_train_val_standardized, scaler_x = apply_standardization(X_train_val_raw)

X_test_standardized = pd.DataFrame(
    scaler_x.transform(X_test_raw),
    columns=X_train_val_raw.columns,
    index=X_test_raw.index
).reset_index(drop=True)

X_train_val_standardized = X_train_val_standardized.reset_index(drop=True)

# 后续所有建模流程统一使用标准化后的 X
X_train_val = X_train_val_standardized
X_test = X_test_standardized


# ======================== 读取 New_sample ========================
# 注意：New_sample 不参与 learning curve 和参数选择，只在最终模型确定后预测。

new_samples_data = pd.read_excel(input_file_X_new_samples)
filtered_new_samples_data = new_samples_data.drop(columns=to_drop)

# 从第三列开始提取特征
# 第二列为空“变形宽度”，各列保持与训练数据一致
X_new_raw = filtered_new_samples_data.iloc[:, 2:]

X_new = pd.DataFrame(
    scaler_x.transform(X_new_raw),
    columns=X_train_val_raw.columns
)

# New_sample 真实值
y_new_true = pd.read_excel(input_file_y_new_true)['真实值']

# New_sample 工程距离约束
Dis_work_new_sample = pd.read_excel(
    input_file_X_new_samples_work_constraint,
    usecols=['工程距离']
)

print(f"\nX_train_val shape: {X_train_val.shape}")
print(f"y_train_val shape: {y_train_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"X_new shape: {X_new.shape}")
print(f"y_new_true shape: {y_new_true.shape}")


# ======================== 采样判定与分箱采样 ========================

def need_sampling(X_train, y_train, std_threshold=0.5, skew_threshold=1.0, kurt_threshold=3.0, lof_neighbors=5):
    """
    判断是否需要对训练样本进行采样增强。
    注意：这里传入的 X_train 已经是标准化后的特征。
    """
    print('train_val 个数为', len(X_train))

    mean_y = np.mean(y_train)
    std_y = np.std(y_train)

    relative_std = std_y / abs(mean_y) if mean_y != 0 else np.inf

    print(f"目标变量均值: {mean_y:.2f}, 标准差: {std_y:.2f}, 相对标准差: {relative_std:.2f}")

    if relative_std > std_threshold:
        print("目标变量相对波动性大，建议进行采样")
        return True

    skew_y = skew(y_train)
    kurt_y = kurtosis(y_train)

    print(f"偏度: {skew_y:.2f}, 峰度: {kurt_y:.2f}")

    if abs(skew_y) > skew_threshold or kurt_y > kurt_threshold:
        print("目标变量偏斜或存在尖峰，建议进行采样")
        return True

    if len(X_train) > 2:
        lof = LocalOutlierFactor(n_neighbors=min(lof_neighbors, len(X_train) - 1))
        outlier_flags = lof.fit_predict(X_train)
        num_outliers = np.sum(outlier_flags == -1)

        if num_outliers > 0:
            print(f"检测到 {num_outliers} 个离群点（LOF），建议进行采样")
            return True

    print("数据分布集中，不需要采样")
    return False


def bin_sampling(X_train_val, y_train_val, min_bin_samples, random_state=None):
    """
    分箱采样。
    注意：X_train_val 为标准化后的特征，因此插值得到的新增样本也处于标准化特征空间。
    y_train_val 保持原始 y 空间。
    """
    if random_state is not None:
        np.random.seed(random_state)

    num_bins = min(6, max(3, int(len(y_train_val) / 10)))
    bins = np.histogram_bin_edges(y_train_val, bins=num_bins)
    y_binned = np.digitize(y_train_val, bins[1:-1], right=True)

    X_resampled, y_resampled = [], []

    is_dataframe = isinstance(X_train_val, pd.DataFrame)
    is_series = isinstance(y_train_val, pd.Series)

    X_np = np.asarray(X_train_val)
    y_np = np.asarray(y_train_val)

    unique_bins, counts_bins = np.unique(y_binned, return_counts=True)

    print(f"原始分箱数: {len(unique_bins)}")
    for b, c in zip(unique_bins, counts_bins):
        print(f"  - Bin {b}: {c} 个样本")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(y_train_val, bins=num_bins, edgecolor='black', alpha=0.7)
    plt.title("原始数据分布")
    plt.xlabel("目标变量值")
    plt.ylabel("样本数")

    for bin_id in np.unique(y_binned):
        mask = y_binned == bin_id
        X_bin = X_np[mask]
        y_bin = y_np[mask]
        n_samples = len(y_bin)

        if n_samples >= min_bin_samples:
            X_resampled.append(X_bin)
            y_resampled.append(y_bin)
        else:
            if n_samples < 2:
                X_interp = np.tile(X_bin, (min_bin_samples, 1))
                y_interp = np.tile(y_bin, min_bin_samples)
            else:
                idx_pairs = [(i, i + 1) for i in range(n_samples - 1)]
                X_interp, y_interp = [], []

                while len(X_interp) + n_samples < min_bin_samples:
                    for i, j in idx_pairs:
                        alpha = np.random.rand()

                        xi = (1 - alpha) * X_bin[i] + alpha * X_bin[j]
                        yi = (1 - alpha) * y_bin[i] + alpha * y_bin[j]

                        X_interp.append(xi)
                        y_interp.append(yi)

                        if len(X_interp) + n_samples >= min_bin_samples:
                            break

                X_interp = np.vstack([X_bin, np.array(X_interp)])
                y_interp = np.concatenate([y_bin, np.array(y_interp)])

            X_resampled.append(X_interp)
            y_resampled.append(y_interp)

    X_resampled = np.vstack(X_resampled)
    y_resampled = np.concatenate(y_resampled)

    y_binned_resampled = np.digitize(y_resampled, bins[1:-1], right=True)
    unique_bins_resampled, counts_resampled = np.unique(y_binned_resampled, return_counts=True)

    print(f"\n重采样后分箱数: {len(unique_bins_resampled)}")
    for b, c in zip(unique_bins_resampled, counts_resampled):
        print(f"  - Bin {b}: {c} 个样本")

    plt.subplot(1, 2, 2)
    plt.hist(y_resampled, bins=num_bins, edgecolor='black', alpha=0.7)
    plt.title("重采样后数据分布")
    plt.xlabel("目标变量值")
    plt.ylabel("样本数")
    plt.tight_layout()

    sample_dist_fig = output_base_path_metrics / 'sampling_distribution.png'
    plt.savefig(sample_dist_fig, dpi=300, bbox_inches='tight')
    plt.close()

    if is_dataframe:
        X_resampled = pd.DataFrame(
            X_resampled,
            columns=X_train_val.columns
        ).reset_index(drop=True)

    if is_series:
        y_resampled = pd.Series(
            y_resampled,
            name=y_train_val.name
        ).reset_index(drop=True)

    return X_resampled, y_resampled


apply_sampling = need_sampling(X_train_val, y_train_val)

if apply_sampling:
    min_bin_samples = 4
    X_train_val, y_train_val = bin_sampling(
        X_train_val,
        y_train_val,
        min_bin_samples,
        random_state=42
    )
else:
    X_train_val = X_train_val.reset_index(drop=True)
    y_train_val = y_train_val.reset_index(drop=True)

# 再次检查采样后的分布
_ = need_sampling(X_train_val, y_train_val)

n_train = int(0.75 * len(X_train_val))


# ======================== 评估指标：原始 y 空间 ========================

def MAE_adjusted(y_train_val, y_test, y_true, y_pred, label='dataset', bin_width=50, save_output=True):
    y_train_val = y_train_val.to_numpy() if isinstance(y_train_val, pd.Series) else np.asarray(y_train_val)
    y_test = y_test.to_numpy() if isinstance(y_test, pd.Series) else np.asarray(y_test)
    y_true = y_true.to_numpy() if isinstance(y_true, pd.Series) else np.asarray(y_true)
    y_pred = y_pred.to_numpy() if isinstance(y_pred, pd.Series) else np.asarray(y_pred)

    y_combined = np.concatenate([y_train_val, y_test])

    alpha = 1.0
    kde = KernelDensity(
        bandwidth=bin_width / 2,
        kernel='gaussian'
    )

    kde.fit(y_combined.reshape(-1, 1))

    density = np.exp(kde.score_samples(y_true.reshape(-1, 1))) + 1e-6
    weights = density ** alpha
    weights /= np.sum(weights)

    mae_adj = np.sum(weights * np.abs(y_true - y_pred))

    if save_output:
        plt.figure()
        plt.scatter(y_true, weights, alpha=1.0)
        plt.xlabel('True Values')
        plt.ylabel('Weights')
        plt.title(f'Weight Distribution vs Target Values ({label})')

        plot_path = output_base_path_weight / f'label_{label}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        df = pd.DataFrame({
            'True Value': y_true.flatten(),
            'Weight': weights
        })

        excel_path = output_base_path_weight / f'label_{label}.xlsx'
        df.to_excel(excel_path, index=False)

    return mae_adj


def smape(y_true, y_pred):
    y_true = y_true.to_numpy() if isinstance(y_true, pd.Series) else np.asarray(y_true)
    y_pred = y_pred.to_numpy() if isinstance(y_pred, pd.Series) else np.asarray(y_pred)

    denominator = (np.abs(y_pred) + np.abs(y_true)) / 2
    denominator = np.where(denominator == 0, 1e-6, denominator)

    return np.mean(np.abs(y_pred - y_true) / denominator) * 100


def smape_adjusted(y_train_val, y_test, y_true, y_pred, bin_width=50):
    y_train_val = y_train_val.to_numpy() if isinstance(y_train_val, pd.Series) else np.asarray(y_train_val)
    y_test = y_test.to_numpy() if isinstance(y_test, pd.Series) else np.asarray(y_test)
    y_true = y_true.to_numpy() if isinstance(y_true, pd.Series) else np.asarray(y_true)
    y_pred = y_pred.to_numpy() if isinstance(y_pred, pd.Series) else np.asarray(y_pred)

    y_combined = np.concatenate([y_train_val, y_test])

    alpha = -1.0
    kde = KernelDensity(
        bandwidth=bin_width / 2,
        kernel='gaussian'
    )

    kde.fit(y_combined.reshape(-1, 1))

    density = np.exp(kde.score_samples(y_true.reshape(-1, 1))) + 1e-6
    weights = density ** alpha
    weights /= np.sum(weights)

    denominator = (np.abs(y_pred) + np.abs(y_true)) / 2
    denominator = np.where(denominator == 0, 1e-6, denominator)

    smape_adj = np.sum(weights * (np.abs(y_pred - y_true) / denominator)) * 100

    return smape_adj


# ======================== 学习曲线 ========================


def plot_learning_curve(
    estimator,
    n_features,
    X_train_val_selected,
    y_train_val,
    X_test_selected,
    y_test,
    param_name,
    param_range,
    cv=4
):
    result_list = []

    for param in param_range:
        estimator.set_params(**{param_name: param})

        kf = KFold(
            n_splits=cv,
            shuffle=True,
            random_state=42
        )

        val_mae_fold = []
        val_mae_adj_fold = []
        val_smape_fold = []
        val_smape_adj_fold = []
        val_r2_fold = []

        train_mae_fold = []
        train_mae_adj_fold = []
        train_smape_fold = []
        train_smape_adj_fold = []
        train_r2_fold = []

        for train_idx, val_idx in kf.split(X_train_val_selected):
            X_train_fold = X_train_val_selected.iloc[train_idx]
            X_val_fold = X_train_val_selected.iloc[val_idx]

            y_train_fold = y_train_val.iloc[train_idx]
            y_val_fold = y_train_val.iloc[val_idx]

            # 每折只用训练折拟合 y 变换器
            pt_fold = fit_y_transformer(y_train_fold)
            y_train_fold_tf = y_transform(pt_fold, y_train_fold)

            estimator.fit(X_train_fold, y_train_fold_tf)

            # 预测后逆变换回原始空间
            y_train_pred_tf = estimator.predict(X_train_fold)
            y_val_pred_tf = estimator.predict(X_val_fold)

            y_train_pred = y_inverse(pt_fold, y_train_pred_tf)
            y_val_pred = y_inverse(pt_fold, y_val_pred_tf)

            train_mae = mean_absolute_error(y_train_fold, y_train_pred)
            val_mae = mean_absolute_error(y_val_fold, y_val_pred)

            train_smape = smape(y_train_fold, y_train_pred)
            val_smape = smape(y_val_fold, y_val_pred)

            train_mae_adj = MAE_adjusted(
                y_train_val,
                y_test,
                y_train_fold,
                y_train_pred,
                label='train',
                save_output=False
            )

            val_mae_adj = MAE_adjusted(
                y_train_val,
                y_test,
                y_val_fold,
                y_val_pred,
                label='val',
                save_output=False
            )

            train_smape_adj = smape_adjusted(
                y_train_val,
                y_test,
                y_train_fold,
                y_train_pred
            )

            val_smape_adj = smape_adjusted(
                y_train_val,
                y_test,
                y_val_fold,
                y_val_pred
            )

            train_r2 = r2_score(y_train_fold, y_train_pred)
            val_r2 = r2_score(y_val_fold, y_val_pred)

            train_mae_fold.append(train_mae)
            val_mae_fold.append(val_mae)

            train_mae_adj_fold.append(train_mae_adj)
            val_mae_adj_fold.append(val_mae_adj)

            train_smape_fold.append(train_smape)
            val_smape_fold.append(val_smape)

            train_smape_adj_fold.append(train_smape_adj)
            val_smape_adj_fold.append(val_smape_adj)

            train_r2_fold.append(train_r2)
            val_r2_fold.append(val_r2)

        # 用全体训练验证集训练，并评估测试集
        pt_all = fit_y_transformer(y_train_val)
        y_train_val_tf = y_transform(pt_all, y_train_val)

        estimator.fit(X_train_val_selected, y_train_val_tf)

        y_test_pred_tf = estimator.predict(X_test_selected)
        y_test_pred = y_inverse(pt_all, y_test_pred_tf)

        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        test_mae_adj = MAE_adjusted(
            y_train_val,
            y_test,
            y_test,
            y_test_pred,
            label='test',
            save_output=False
        )

        test_smape = smape(y_test, y_test_pred)

        test_smape_adj = smape_adjusted(
            y_train_val,
            y_test,
            y_test,
            y_test_pred
        )

        result_list.append({
            'param': param,

            'train_mae': np.mean(train_mae_fold),
            'val_mae': np.mean(val_mae_fold),
            'test_mae': test_mae,

            'train_smape': np.mean(train_smape_fold),
            'val_smape': np.mean(val_smape_fold),
            'test_smape': test_smape,

            'train_mae_adjusted': np.mean(train_mae_adj_fold),
            'val_mae_adjusted': np.mean(val_mae_adj_fold),
            'test_mae_adjusted': test_mae_adj,

            'train_smape_adjusted': np.mean(train_smape_adj_fold),
            'val_smape_adjusted': np.mean(val_smape_adj_fold),
            'test_smape_adjusted': test_smape_adj,

            'train_r2': np.mean(train_r2_fold),
            'val_r2': np.mean(val_r2_fold),
            'test_r2': test_r2,
        })

    results_df = pd.DataFrame(result_list)

    # MAE 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, results_df['train_mae'], label='Train MAE', marker='o')
    plt.plot(param_range, results_df['val_mae'], label='Val MAE', marker='x')
    plt.plot(param_range, results_df['test_mae'], label='Test MAE', marker='s')
    plt.xlabel(param_name)
    plt.ylabel('MAE')
    plt.title(f'Learning Curve - n_features: {n_features}')
    plt.legend()
    plt.grid(True)

    save_path = output_base_path_curves_mae / f'n_features_{n_features}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # SMAPE 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, results_df['train_smape'], label='Train SMAPE', marker='o')
    plt.plot(param_range, results_df['val_smape'], label='Val SMAPE', marker='x')
    plt.plot(param_range, results_df['test_smape'], label='Test SMAPE', marker='s')
    plt.xlabel(param_name)
    plt.ylabel('SMAPE')
    plt.title(f'Learning Curve - n_features: {n_features}')
    plt.legend()
    plt.grid(True)

    save_path = output_base_path_curves_smape / f'n_features_{n_features}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # MAE_adjusted 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, results_df['train_mae_adjusted'], label='Train MAE_adjusted', marker='o')
    plt.plot(param_range, results_df['val_mae_adjusted'], label='Val MAE_adjusted', marker='x')
    plt.plot(param_range, results_df['test_mae_adjusted'], label='Test MAE_adjusted', marker='s')
    plt.xlabel(param_name)
    plt.ylabel('MAE_adjusted')
    plt.title(f'Learning Curve - n_features: {n_features}')
    plt.legend()
    plt.grid(True)

    save_path = output_base_path_curves_mae_adjusted / f'n_features_{n_features}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # SMAPE_adjusted 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, results_df['train_smape_adjusted'], label='Train SMAPE_adjusted', marker='o')
    plt.plot(param_range, results_df['val_smape_adjusted'], label='Val SMAPE_adjusted', marker='x')
    plt.plot(param_range, results_df['test_smape_adjusted'], label='Test SMAPE_adjusted', marker='s')
    plt.xlabel(param_name)
    plt.ylabel('SMAPE_adjusted')
    plt.title(f'Learning Curve - n_features: {n_features}')
    plt.legend()
    plt.grid(True)

    save_path = output_base_path_curves_smape_adjusted / f'n_features_{n_features}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return results_df


# ======================== 学习曲线与调参 ========================

all_results = {}
final_sorted_indices = None

metrics_to_save = {
    "train_mae": "train_mae_results_KNN_reg.xlsx",
    "train_smape": "train_smape_results_KNN_reg.xlsx",
    "train_mae_adjusted": "train_mae_adjusted_results_KNN_reg.xlsx",
    "train_smape_adjusted": "train_smape_adjusted_results_KNN_reg.xlsx",
    "train_r2": "train_r2_results_KNN_reg.xlsx",

    "val_mae": "val_mae_results_KNN_reg.xlsx",
    "val_smape": "val_smape_results_KNN_reg.xlsx",
    "val_mae_adjusted": "val_mae_adjusted_results_KNN_reg.xlsx",
    "val_smape_adjusted": "val_smape_adjusted_results_KNN_reg.xlsx",
    "val_r2": "val_r2_results_KNN_reg.xlsx",

    "test_mae": "test_mae_results_KNN_reg.xlsx",
    "test_smape": "test_smape_results_KNN_reg.xlsx",
    "test_mae_adjusted": "test_mae_adjusted_results_KNN_reg.xlsx",
    "test_smape_adjusted": "test_smape_adjusted_results_KNN_reg.xlsx",
    "test_r2": "test_r2_results_KNN_reg.xlsx",
}

for selected_feature_count in range(1, 28):

    print("\nnumber of the selected feature names:", selected_feature_count)

    X_input = X_train_val
    y_input = y_train_val

    # Lasso 在标准化后的 X 上进行特征排序
    lasso = Lasso(
        alpha=0.1,
        random_state=42,
        max_iter=15000
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

    knn = KNeighborsRegressor(
        metric='manhattan',
        weights='distance'
    )

    param_range = range(1, n_train)

    results_df = plot_learning_curve(
        estimator=knn,
        n_features=selected_feature_count,
        X_train_val_selected=X_train_val_selected,
        y_train_val=y_train_val,
        X_test_selected=X_test_selected,
        y_test=y_test,
        param_name='n_neighbors',
        param_range=param_range,
        cv=4
    )

    all_results[selected_feature_count] = results_df

    for metric_name, file_name in metrics_to_save.items():
        metric_df = pd.DataFrame()

        for feature_count, temp_results_df in all_results.items():
            temp_df = temp_results_df[['param', metric_name]].copy()
            temp_df = temp_df.set_index('param').T
            temp_df.index = [f"Features_{feature_count}"]
            metric_df = pd.concat([metric_df, temp_df])

        metric_df = metric_df.T

        output_path = output_base_path_metrics / file_name
        metric_df.to_excel(output_path)


# ======================== 确定最终参数 ========================

final_n_features = 11
n_neighbors = 1

if final_sorted_indices is None:
    raise ValueError("Lasso 特征排序结果为空，请检查输入数据。")

best_selected_features = final_sorted_indices[:final_n_features]

print("\n最终选择的特征索引：")
print(best_selected_features)

print("\n最终选择的特征名称：")
print(list(X_train_val.columns[best_selected_features]))

X_train_val_final = X_train_val.iloc[:, best_selected_features]
X_test_final = X_test.iloc[:, best_selected_features]

knn = KNeighborsRegressor(
    n_neighbors=n_neighbors,
    metric='manhattan',
    weights='distance'
)


# ======================== 4 折交叉验证 ========================

cv_detail_rows = []

kf = KFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

train_mae_scores = []
train_smape_scores = []
train_r2_scores = []
train_mae_adj_scores = []
train_smape_adj_scores = []

val_mae_scores = []
val_smape_scores = []
val_r2_scores = []
val_mae_adj_scores = []
val_smape_adj_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val_final)):

    X_train_final = X_train_val_final.iloc[train_idx]
    X_val_final = X_train_val_final.iloc[val_idx]

    y_train = y_train_val.iloc[train_idx]
    y_val = y_train_val.iloc[val_idx]

    # 该折拟合 y 变换器，并在变换空间训练
    pt_fold = fit_y_transformer(y_train)
    y_train_tf = y_transform(pt_fold, y_train)

    knn.fit(X_train_final, y_train_tf)

    # 预测后逆变换回原始空间
    y_train_pred_tf = knn.predict(X_train_final)
    y_val_pred_tf = knn.predict(X_val_final)

    y_train_pred = y_inverse(pt_fold, y_train_pred_tf)
    y_val_pred = y_inverse(pt_fold, y_val_pred_tf)

    # 训练集评估
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_smape = smape(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    train_mae_adj = MAE_adjusted(
        y_train_val,
        y_test,
        y_train,
        y_train_pred,
        f'Train fold {fold + 1}',
        save_output=True
    )

    train_smape_adj = smape_adjusted(
        y_train_val,
        y_test,
        y_train,
        y_train_pred
    )

    train_mae_scores.append(train_mae)
    train_smape_scores.append(train_smape)
    train_r2_scores.append(train_r2)
    train_mae_adj_scores.append(train_mae_adj)
    train_smape_adj_scores.append(train_smape_adj)

    # 验证集评估
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_smape = smape(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    val_mae_adj = MAE_adjusted(
        y_train_val,
        y_test,
        y_val,
        y_val_pred,
        f'Validation fold {fold + 1}',
        save_output=True
    )

    val_smape_adj = smape_adjusted(
        y_train_val,
        y_test,
        y_val,
        y_val_pred
    )

    val_mae_scores.append(val_mae)
    val_smape_scores.append(val_smape)
    val_r2_scores.append(val_r2)
    val_mae_adj_scores.append(val_mae_adj)
    val_smape_adj_scores.append(val_smape_adj)

    print(f"\nFold {fold + 1} Results:")
    print(
        f"Train MAE: {train_mae:.4f}, "
        f"R2: {train_r2:.4f}, "
        f"MAE_adj: {train_mae_adj:.4f}, "
        f"SMAPE_adj: {train_smape_adj:.4f}"
    )

    print(
        f"Val MAE: {val_mae:.4f}, "
        f"R2: {val_r2:.4f}, "
        f"MAE_adj: {val_mae_adj:.4f}, "
        f"SMAPE_adj: {val_smape_adj:.4f}"
    )

    print("-" * 40)

    train_row_ids = X_train_val_final.index[train_idx]
    val_row_ids = X_train_val_final.index[val_idx]

    for rid, yy, pp in zip(train_row_ids, y_train, y_train_pred):
        cv_detail_rows.append({
            "fold": fold + 1,
            "set": "train",
            "row_id": int(rid),
            "y_true": float(yy),
            "y_pred": float(pp)
        })

    for rid, yy, pp in zip(val_row_ids, y_val, y_val_pred):
        cv_detail_rows.append({
            "fold": fold + 1,
            "set": "val",
            "row_id": int(rid),
            "y_true": float(yy),
            "y_pred": float(pp)
        })


# ======================== 输出交叉验证平均结果 ========================

print("\n输出交叉验证的平均结果（Cross-Validation Average Results）:")

print(f"训练集 MAE: {np.mean(train_mae_scores):.4f}")
print(f"训练集 SMAPE: {np.mean(train_smape_scores):.4f}")
print(f"训练集 R2: {np.mean(train_r2_scores):.4f}")
print(f"训练集 MAE_adj: {np.mean(train_mae_adj_scores):.4f}")
print(f"训练集 SMAPE_adj: {np.mean(train_smape_adj_scores):.4f}")

print(f"验证集 MAE: {np.mean(val_mae_scores):.4f}")
print(f"验证集 SMAPE: {np.mean(val_smape_scores):.4f}")
print(f"验证集 R2: {np.mean(val_r2_scores):.4f}")
print(f"验证集 MAE_adj: {np.mean(val_mae_adj_scores):.4f}")
print(f"验证集 SMAPE_adj: {np.mean(val_smape_adj_scores):.4f}")


# ======================== 保存交叉验证逐样本结果 ========================

cv_detail_df = pd.DataFrame(cv_detail_rows)

cv_mean_df = (
    cv_detail_df
    .groupby("row_id", as_index=False)
    .agg(
        y_true=("y_true", "first"),
        y_pred_mean=("y_pred", "mean"),
        n_preds=("y_pred", "size")
    )
)

print("每个样本参与的预测次数分布：", cv_mean_df["n_preds"].value_counts().to_dict())

out_cv_detail_path = output_base_path_model / 'CV_true_pred_by_fold.xlsx'

with pd.ExcelWriter(out_cv_detail_path, engine='xlsxwriter') as writer:
    cv_detail_df.to_excel(
        writer,
        sheet_name='cv_true_pred',
        index=False
    )

    cv_mean_df[["row_id", "y_true", "y_pred_mean"]].to_excel(
        writer,
        sheet_name='cv_mean_per_sample',
        index=False
    )

print(f"交叉验证逐样本与均值结果已保存到：{out_cv_detail_path}")


# ======================== 最终模型：用全训练集拟合 y 变换器并训练 ========================

pt_all = fit_y_transformer(y_train_val)
y_train_val_tf = y_transform(pt_all, y_train_val)

knn.fit(X_train_val_final, y_train_val_tf)


# ======================== 测试集评估 ========================

y_test_pred_tf = knn.predict(X_test_final)
y_test_pred = y_inverse(pt_all, y_test_pred_tf)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_smape = smape(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

test_mae_adj = MAE_adjusted(
    y_train_val,
    y_test,
    y_test,
    y_test_pred,
    'Test',
    save_output=True
)

test_smape_adj = smape_adjusted(
    y_train_val,
    y_test,
    y_test,
    y_test_pred
)

print("\n模型表现评估（Final Model Performance）:")
print(f"测试集 MAE: {test_mae:.4f}")
print(f"测试集 SMAPE: {test_smape:.4f}")
print(f"测试集 R2: {test_r2:.4f}")
print(f"测试集 MAE_adj: {test_mae_adj:.4f}")
print(f"测试集 SMAPE_adj: {test_smape_adj:.4f}")

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


# ======================== 保存模型、标准化器、y 变换器、最终特征索引 ========================

model_path = output_base_path_model / 'model_KNN_r_202505.pkl'
dump(knn, model_path)
print(f"\n模型已保存至: {model_path}")

scalerX_path = output_base_path_model / 'scalerX_KNN_r_202505.pkl'
joblib.dump(scaler_x, scalerX_path)
print(f"X 标准化器已保存至: {scalerX_path}")

y_transformer_path = output_base_path_model / 'y_power_transformer_202505.pkl'
joblib.dump(pt_all, y_transformer_path)
print(f"y 变换器已保存至: {y_transformer_path}")

selected_features_path = output_base_path_model / 'selected_feature_indices_reg.pkl'
joblib.dump(best_selected_features, selected_features_path)
print(f"最终特征索引已保存至: {selected_features_path}")


# ======================== 加载模型和预处理器 ========================

best_model = joblib.load(model_path)
scaler_x = joblib.load(scalerX_path)
pt_loaded = joblib.load(y_transformer_path)
best_selected_features = joblib.load(selected_features_path)


# ======================== New_sample 最终预测 ========================
# 注意：
# 1. New_sample 不参与调参；
# 2. New_sample 已经用训练集 scaler_x 标准化；
# 3. 这里只使用最终确定的特征和最终模型进行预测；
# 4. 模型输出先在 y 变换空间，随后逆变换回原始 y 空间。

def evaluate_regression(y_true, y_pred):
    y_true = y_true.to_numpy() if isinstance(y_true, pd.Series) else np.asarray(y_true)
    y_pred = y_pred.to_numpy() if isinstance(y_pred, pd.Series) else np.asarray(y_pred)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator == 0, 1e-6, denominator)

    smape_v = np.mean(
        2 * np.abs(y_true - y_pred) / denominator
    ) * 100

    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"SMAPE: {smape_v:.2f}%")

    return r2, mae, smape_v


def predict_new_samples_regression(X_new, y_new_true):
    all_feature_names = X_new.columns.tolist()

    X_new_sel = X_new.iloc[:, best_selected_features]

    X_new_feature_names = [
        all_feature_names[i] for i in best_selected_features
    ]

    print("\n预测 New_sample 时使用的特征名称：")
    print(', '.join(X_new_feature_names))

    # New_sample 预测：变换空间 → 原始空间
    y_pred_tf = best_model.predict(X_new_sel)
    y_pred = y_inverse(pt_loaded, y_pred_tf)

    # 工程距离约束
    work_distance_constraint = -Dis_work_new_sample['工程距离'].to_numpy()
    y_pred = np.maximum(work_distance_constraint, y_pred)

    print()
    print('New_sample 真实值:')
    print(y_new_true)

    print('New_sample 预测值:')
    print(y_pred)

    r2, mae, smape_v = evaluate_regression(y_new_true, y_pred)

    new_sample_results = pd.DataFrame({
        '真实值': y_new_true,
        '预测值': y_pred,
        '工程距离约束值': work_distance_constraint
    })

    output_new_sample_file = output_base_path_metrics / 'new_sample_prediction_results.xlsx'
    new_sample_results.to_excel(output_new_sample_file, index=False)

    print(f"New_sample 预测结果已保存至: {output_new_sample_file}")

    return r2, mae, smape_v, new_sample_results


predict_new_samples_regression(X_new, y_new_true)
