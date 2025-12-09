import scipy.linalg
import numpy as np


class CSP:
    def __init__(self, m_filters):
        self.m_filters = m_filters

    def fit(self, x_train, y_train):
        x_data = np.copy(x_train)
        y_labels = np.copy(y_train)
        n_trials, n_channels, n_samples = x_data.shape  # 分别为试验次数、脑电信号通道数、采样点数
        cov_x = np.zeros((2, n_channels, n_channels), dtype=np.float64)  # 初始化元素为零的三维数组大小（2，n，n），cov_x用于储存协方差矩阵
        for i in range(n_trials):  # 每个试验单独处理
            x_trial = x_data[i, :, :]  # 依次提取元素
            y_trial = y_labels[i]  # 标签值0，1等，用于指定类别
            cov_x_trial = np.matmul(x_trial, np.transpose(x_trial))  # matmul为向量点积、矩阵乘法，transpose即转置，x_trail的后两个维度（二维矩阵）与x_trail的转置做矩阵乘法
            cov_x_trial /= np.trace(cov_x_trial)  # 除矩阵的迹，得协方差矩阵
            cov_x[y_trial, :, :] += cov_x_trial  # 将对应的类别的协方差矩阵相加，[:, :]用于选择整个维度的所有元素。

        cov_x = np.asarray([cov_x[cls] / np.sum(y_labels == cls) for cls in range(2)])  ##归一化操作
        cov_combined = cov_x[0] + cov_x[1]
        eig_values, u_mat = scipy.linalg.eig(cov_combined, cov_x[0])  ##计算了 cov_combined 矩阵的特征值和特征向量。特征值保存在 eig_values 中，
                                                                       # 而特征向量保存在 u_mat 中。通常，特征值表示了矩阵在特征向量方向上的变化程度。
        sort_indices = np.argsort(abs(eig_values))[::-1]  ##算了特征值的绝对值，并且根据特征值的大小进行降序排列，得到排序后的索引 sort_indices
        eig_values = eig_values[sort_indices]
        u_mat = u_mat[:, sort_indices]  ##对特征值和特征向量进行重新排序，以便它们按照特征值的大小降序排列。
        u_mat = np.transpose(u_mat)  ##转置
        return eig_values, u_mat

    def transform(self, x_trial, eig_vectors):
        z_trial = np.matmul(eig_vectors, x_trial)
        z_trial_selected = z_trial[:self.m_filters, :]
        z_trial_selected = np.append(z_trial_selected, z_trial[-self.m_filters:, :],axis=0)  ##在 z_trial_selected 矩阵的末尾添加 z_trial 矩阵的最后 self.m_filters 行
        sum_z2 = np.sum(z_trial_selected ** 2, axis=1)  ##平方和
        sum_z = np.sum(z_trial_selected, axis=1)  ##元素和
        var_z = (sum_z2 - (sum_z ** 2) / z_trial_selected.shape[1]) / (z_trial_selected.shape[1] - 1)  ##方差
        sum_var_z = sum(var_z)
        return np.log(var_z / sum_var_z)  ##特征向量，一维

    def transform_loop(self,x_trial,eig_vectors):
        n_trials = x_trial.shape[0]
        # print(f'n_trials is {n_trials}')
        x_features = np.zeros((n_trials, 4), dtype=np.float64)
        for k in range(n_trials):
            trial = np.copy(x_trial[ k, :, :])
            csp_feat = self.transform(trial, eig_vectors)
            for j in range(self.m_filters):
                x_features[k , (j + 1) * 2 - 2] = csp_feat[j]
                x_features[k , (j + 1) * 2 - 1] = csp_feat[-j - 1]
        return x_features

