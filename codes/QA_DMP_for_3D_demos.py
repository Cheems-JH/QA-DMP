###########################################################
# Learning 3D multi-quality demos (X,Y,Z) with DMP and QA-DMP
###########################################################
import numpy as np
from ts2vec import TS2Vec
import matplotlib.pyplot as plt
from dtw import *
from code_dataset_pub.code import GMM_DMP
import os
import torch
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import solve_banded
from itertools import combinations


# Key parameters
alpha_m = 0.5 # Specify the concentration degree of score allocation for the DTW-based identification
alpha_n = 0.5 # Specify the concentration degree of score allocation for the TS2Vec-based identification
lambda_c = 0.5 # Specify the degree of emphasis on the fusion of DTW-based and TS2Vec-based identification results, usually =0.5, indicating that both methods are taken into account
alpha_c = 0.8 # Specify the concentration degree of score allocation for the identification results after fusion
lambda_s = 50 # Specify the degree of smoothing penalty during QA-DMP learning
cnt_sample=10 # Specify the quantity of DMP and QA-DMP outputs


print("---------------STEP-1: Loading Demos--------------")
all_demos = np.load('~/dataset/3D_simulation_demos.npy', allow_pickle=True).tolist()
# 读取x轴和y轴和z轴的演示和基准
baseline_x = []
baseline_y = []
baseline_z = []
y_demos = []
x_demos = []
z_demos = []
for i, demo in enumerate(all_demos):
    x = np.array(demo['x'])
    y = np.array(demo['y'])
    z = np.array(demo['z'])
    if i == 0:
        baseline_x = x
        baseline_y = y
        baseline_z = z
    else:
        x_demos.append(x)
        y_demos.append(y)
        z_demos.append(z)
N_demos = len(x_demos)
L_demos = len(x_demos[0])
t = np.linspace(1, L_demos, L_demos) # 时间索引
print('Number of demos:', N_demos)
print('Length of each demo', L_demos)
x_ts2vec_model_path = '~/dataset/2D_simulation_demos_ts2vec_x.pth'
y_ts2vec_model_path = '~/dataset/2D_simulation_demos_ts2vec_y.pth'
z_ts2vec_model_path = '~/dataset/2D_simulation_demos_ts2vec_z.pth'


print("---------------STEP-2: Pre-processing Demos--------------")
x_demos_space1 = x_demos
y_demos_space1 = y_demos
z_demos_space1 = z_demos
print('Generating forcing-view dataset')
x_demos_space2 = []
y_demos_space2 = []
z_demos_space2 = []
for i in range(N_demos):
     gmm_dmp_x = GMM_DMP.EmGmmDMP(data_set=x_demos[i], n_rbf=10) # 这里的n_rbf无所谓,因为只需要强迫项
     gmm_dmp_y = GMM_DMP.EmGmmDMP(data_set=y_demos[i], n_rbf=10)
     gmm_dmp_z = GMM_DMP.EmGmmDMP(data_set=z_demos[i], n_rbf=10)
     f_target_x = gmm_dmp_x.bm_f_target
     x_demos_space2.append(f_target_x)
     f_target_y = gmm_dmp_y.bm_f_target
     y_demos_space2.append(f_target_y)
     f_target_z = gmm_dmp_z.bm_f_target
     z_demos_space2.append(f_target_z)
# 辅助材料：最优演示对应的强迫项
baseline_gmm_dmp_x = GMM_DMP.EmGmmDMP(data_set=baseline_x, n_rbf=10)
baseline_gmm_dmp_y = GMM_DMP.EmGmmDMP(data_set=baseline_y, n_rbf=10)
baseline_gmm_dmp_z = GMM_DMP.EmGmmDMP(data_set=baseline_z, n_rbf=10)
baseline_f_target_x = baseline_gmm_dmp_x.bm_f_target
baseline_f_target_y = baseline_gmm_dmp_y.bm_f_target
baseline_f_target_z = baseline_gmm_dmp_z.bm_f_target
bm_s_x = baseline_gmm_dmp_x.bm_s
bm_s_y = baseline_gmm_dmp_y.bm_s
bm_s_z = baseline_gmm_dmp_z.bm_s


print("---------------STEP-3: Relative Quality Identification and Score Assigning --------------")
# Functions for DTW-based identification
def calculate_global_avg_dtw(sequences, num_sequences):
    sequences = sequences
    N = num_sequences
    dtw_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            alignment = dtw(sequences[i], sequences[j], keep_internals=True)
            dist = alignment.distance
            dtw_matrix[i][j] = dist
            dtw_matrix[j][i] = dist
    avg_dtw = []
    for i in range(N):
        avg = (np.sum(dtw_matrix[i]) - dtw_matrix[i][i]) / (N - 1)
        avg_dtw.append(avg)
    return avg_dtw
def calculate_weights_based_global_avg_dtw(avg_dtw, length_sequences, alpha=0.5):
    avg_dtw = (avg_dtw - np.mean(avg_dtw)) / np.std(avg_dtw)
    global_scores = np.array(avg_dtw)
    scores = -global_scores
    scaled = alpha * scores
    scaled -= np.max(scaled)
    global_weights = np.exp(scaled)
    global_weights /= np.sum(global_weights)
    global_weights_per_point = np.repeat(global_weights[:, np.newaxis], length_sequences, axis=1)
    point_weights = global_weights_per_point.flatten()
    return point_weights

# Functions for TS2Vec-based identification
def smooth_embeddings(embeddings, sigma=10.0):
    N, T, D = embeddings.shape
    smoothed = np.empty_like(embeddings)
    for n in range(N):
        for d in range(D):
            smoothed[n, :, d] = gaussian_filter1d(embeddings[n, :, d], sigma=sigma, mode='nearest')
    return smoothed
def calculated_local_qualities_based_only_ts2vec(training_dataset, model, alpha=0.5):
    local_embeddings = model.encode(training_dataset, sliding_length=1)
    smoothed_embeddings = smooth_embeddings(local_embeddings)
    local_embeddings = smoothed_embeddings
    mean_embeddings = np.mean(local_embeddings, axis=0)
    diffs = local_embeddings - mean_embeddings[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    mean_dists = np.mean(dists, axis=0, keepdims=True)    # (1, T)
    std_dists = np.std(dists, axis=0, keepdims=True) + 1e-8
    dists_normalized = (dists - mean_dists) / std_dists    # (N, T)
    scaled = -alpha * dists_normalized
    scaled -= np.max(scaled, axis=0, keepdims=True)
    weights_per_t = np.exp(scaled)
    weights_per_t /= np.sum(weights_per_t, axis=0, keepdims=True)  # (N, T)
    weights = weights_per_t.flatten()
    return weights

# Functions for fusion of DTW-based and TS2Vec-based identification results
def calculated_combined_weights(global_weights, local_weights, combined_coff=0.5):
    combined_weights = np.power(global_weights, combined_coff) * np.power(local_weights, 1 - combined_coff)
    combined_weights /= np.sum(combined_weights)
    return combined_weights
def timewise_normalized_softmax(weights_2d, alpha=5.0, eps=1e-66):
    mean_per_time = np.mean(weights_2d, axis=0, keepdims=True)
    std_per_time = np.std(weights_2d, axis=0, keepdims=True) + eps
    weights_normalized = (weights_2d - mean_per_time) / std_per_time
    scaled = alpha * weights_normalized
    scaled -= np.max(scaled, axis=0, keepdims=True)
    exp_scaled = np.exp(scaled)
    softmaxed = exp_scaled / np.sum(exp_scaled, axis=0, keepdims=True)
    max_per_time = np.max(softmaxed, axis=0, keepdims=True)
    normalized = softmaxed / (max_per_time + eps)
    return normalized

# Training ts2vec models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_space2_ts2vec_dataset = np.array([seq[:, np.newaxis] for seq in x_demos_space2])
x_space2_ts2vec_model = TS2Vec(input_dims=1, device=device)
if os.path.exists(x_ts2vec_model_path):
    x_space2_ts2vec_model.load(x_ts2vec_model_path)
    print("Saved X TS2Vec model loaded.")
else:
    print("Training X TS2Vec model...")
    x_space2_ts2vec_model.fit(x_space2_ts2vec_dataset, n_epochs=200, n_iters=200, verbose=True)
    x_space2_ts2vec_model.save(x_ts2vec_model_path)
    print("X TS2Vec Model saved to:", x_ts2vec_model_path)
y_space2_ts2vec_dataset = np.array([seq[:, np.newaxis] for seq in y_demos_space2])
y_space2_ts2vec_model = TS2Vec(input_dims=1, device=device)
if os.path.exists(y_ts2vec_model_path): # 如果模型已保存，则加载；否则训练并保存
    y_space2_ts2vec_model.load(y_ts2vec_model_path)
    print("Saved Y TS2Vec model loaded.")
else:
    print("Training Y TS2Vec model...")
    y_space2_ts2vec_model.fit(y_space2_ts2vec_dataset, n_epochs=200, n_iters=200, verbose=True)
    y_space2_ts2vec_model.save(y_ts2vec_model_path)
    print("Y TS2Vec Model saved to:", y_ts2vec_model_path)
# 训练z维度的ts2vec模型
z_space2_ts2vec_dataset = np.array([seq[:, np.newaxis] for seq in z_demos_space2])
z_space2_ts2vec_model = TS2Vec(input_dims=1, device=device)
if os.path.exists(z_ts2vec_model_path): # 如果模型已保存，则加载；否则训练并保存
    z_space2_ts2vec_model.load(z_ts2vec_model_path)
    print("Saved Z TS2Vec model loaded.")
else:
    print("Training Z TS2Vec model...")
    z_space2_ts2vec_model.fit(z_space2_ts2vec_dataset, n_epochs=200, n_iters=200, verbose=True)
    z_space2_ts2vec_model.save(z_ts2vec_model_path)
    print("Z TS2Vec model saved to:", z_ts2vec_model_path)

# Quality identification
print('DTW-based quality identification...')
x_space2_avg_dtw = calculate_global_avg_dtw(sequences=x_demos_space2, num_sequences=N_demos)
x_space2_global_weights = calculate_weights_based_global_avg_dtw(x_space2_avg_dtw, length_sequences=L_demos, alpha=alpha_m)
y_space2_avg_dtw = calculate_global_avg_dtw(sequences=y_demos_space2, num_sequences=N_demos)
y_space2_global_weights = calculate_weights_based_global_avg_dtw(y_space2_avg_dtw, length_sequences=L_demos, alpha=alpha_m)
z_space2_avg_dtw = calculate_global_avg_dtw(sequences=z_demos_space2, num_sequences=N_demos)
z_space2_global_weights = calculate_weights_based_global_avg_dtw(z_space2_avg_dtw, length_sequences=L_demos, alpha=alpha_m)

print('TS2Vec-based quality identification...')
x_space2_local_weights = calculated_local_qualities_based_only_ts2vec(x_space2_ts2vec_dataset, x_space2_ts2vec_model, alpha=alpha_n)
y_space2_local_weights = calculated_local_qualities_based_only_ts2vec(y_space2_ts2vec_dataset, y_space2_ts2vec_model, alpha=alpha_n)
z_space2_local_weights = calculated_local_qualities_based_only_ts2vec(z_space2_ts2vec_dataset, z_space2_ts2vec_model, alpha=alpha_n)

print('Fusion of identification results...')
x_space2_combined_weights = calculated_combined_weights(x_space2_global_weights, x_space2_local_weights, combined_coff=lambda_c)
y_space2_combined_weights = calculated_combined_weights(y_space2_global_weights, y_space2_local_weights, combined_coff=lambda_c)
z_space2_combined_weights = calculated_combined_weights(z_space2_global_weights, z_space2_local_weights, combined_coff=lambda_c)
x_space2_weights_2d = x_space2_combined_weights.reshape((N_demos, L_demos))
y_space2_weights_2d = y_space2_combined_weights.reshape((N_demos, L_demos))
z_space2_weights_2d = z_space2_combined_weights.reshape((N_demos, L_demos))
x_space2_soft_weights_2d = timewise_normalized_softmax(x_space2_weights_2d, alpha_c)
x_space2_final_weights = x_space2_soft_weights_2d.flatten()
x_space2_final_weights /= np.sum(x_space2_final_weights)
y_space2_soft_weights_2d = timewise_normalized_softmax(y_space2_weights_2d, alpha_c)
y_space2_final_weights = y_space2_soft_weights_2d.flatten()
y_space2_final_weights /= np.sum(y_space2_final_weights)
z_space2_soft_weights_2d = timewise_normalized_softmax(z_space2_weights_2d, alpha_c)
z_space2_final_weights = z_space2_soft_weights_2d.flatten()
z_space2_final_weights /= np.sum(z_space2_final_weights)



print("---------------STEP-4: generation outputs by DMP and QA-DMP--------------")
# Functions for dmp generation
def weighted_gmm_reproduction(weighted_gmm, bm_s, start=None, target=None):
    tau = 1.0
    alpha_x = 1.0
    alpha_y = 60.0
    beta_y = 60.0 / 4.0
    g = target
    y0 = start
    means = weighted_gmm.means_
    covariances = weighted_gmm.covariances_
    weights = weighted_gmm.weights_gmm
    sampled_bm_f_target = np.zeros_like(bm_s)

    for i, s in enumerate(bm_s):
        conditional_means = []
        conditional_covs = []
        conditional_weights = []
        for k in range(weighted_gmm.n_components):
            mu_k = means[k]
            Sigma_k = covariances[k]
            mu_s_k = mu_k[0]
            mu_f_k = mu_k[1]
            Sigma_ss_k = Sigma_k[0, 0]
            Sigma_sf_k = Sigma_k[0, 1]
            Sigma_fs_k = Sigma_k[1, 0]
            Sigma_ff_k = Sigma_k[1, 1]
            conditional_mu_f_k = mu_f_k + (Sigma_fs_k / Sigma_ss_k) * (s - mu_s_k)
            conditional_Sigma_f_k = Sigma_ff_k - (Sigma_fs_k / Sigma_ss_k) * Sigma_sf_k
            conditional_means.append(conditional_mu_f_k)
            conditional_covs.append(conditional_Sigma_f_k)
            likelihood_s = scipy.stats.norm.pdf(s, loc=mu_s_k, scale=np.sqrt(Sigma_ss_k))
            conditional_weights.append(weights[k] * likelihood_s)
        conditional_weights = np.array(conditional_weights)
        conditional_weights /= np.sum(conditional_weights)
        component = np.random.choice(weighted_gmm.n_components, p=conditional_weights)
        sampled_bm_f_target[i] = np.random.normal(conditional_means[component], np.sqrt(conditional_covs[component]))
    bm_s_generate = bm_s
    bm_f_target_generate = sampled_bm_f_target
    x_g = np.array(bm_s_generate) / (g - y0)
    t_g = np.zeros_like(x_g)
    for k in range(len(bm_s_generate) - 1):
        delta_k = (x_g[k + 1] - x_g[k]) / tau
        t_g[k] = delta_k / (- alpha_x * x_g[k])
    y_reproduce = np.zeros(len(bm_s_generate))
    dy_reproduce = np.zeros(len(bm_s_generate))
    ddy_reproduce = np.zeros(len(bm_s_generate))
    tmp_y = start
    tmp_dy = 0.0
    tmp_ddy = 0.0
    gain = (target - start) / (g - y0)
    for k in range(len(bm_s_generate)):
        y_reproduce[k] = tmp_y
        dy_reproduce[k] = tmp_dy
        ddy_reproduce[k] = tmp_ddy
        tmp_ddy = alpha_y * (beta_y * (target - tmp_y) - tmp_dy) + bm_f_target_generate[k] * gain
        tmp_dy += tau * tmp_ddy * t_g[k]
        tmp_y += tau * tmp_dy * t_g[k]
    return y_reproduce, sampled_bm_f_target

# Class for GMM and its EM training
class VanillaGMM:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
    def fit(self, X):
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        global_cov = np.cov(X.T) if n_features > 1 else np.var(X)
        self.covariances_ = [np.atleast_2d(global_cov) + np.eye(n_features) * 1e-6 for _ in range(self.n_components)]
        self.weights_gmm = np.ones(self.n_components) / self.n_components

        for iteration in range(self.max_iter):
            # ---------- E-step ----------
            log_prob = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                try:
                    log_prob[:, k] = np.log(self.weights_gmm[k] + 1e-8) + multivariate_normal.logpdf(
                        X,
                        mean=np.atleast_1d(self.means_[k]),
                        cov=np.atleast_2d(self.covariances_[k]),
                        allow_singular=True
                    )
                except np.linalg.LinAlgError as e:
                    print(f"[WARNING] Component {k} covariance invalid, skipping. Error: {e}")
                    log_prob[:, k] = -np.inf

            log_prob_norm = logsumexp(log_prob, axis=1, keepdims=True)
            resp = np.exp(log_prob - log_prob_norm)

            # ---------- M-step ----------
            Nk = resp.sum(axis=0) + 1e-8
            self.weights_gmm = Nk / np.sum(Nk)
            self.means_ = (resp.T @ X) / Nk[:, np.newaxis]
            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted_diff = diff.T * resp[:, k]
                cov_k = (weighted_diff @ diff) / Nk[k]
                self.covariances_[k] = np.atleast_2d(cov_k) + np.eye(n_features) * 1e-6

            # ---------- Check convergence ----------
            if iteration > 0:
                delta = np.linalg.norm(self.means_ - prev_means)
                if delta < self.tol:
                    print(f"Converged at iteration {iteration}")
                    break
            prev_means = np.copy(self.means_)

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        log_prob = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            log_prob[:, k] = np.log(self.weights_gmm[k] + 1e-8) + multivariate_normal.logpdf(
                X,
                mean=np.atleast_1d(self.means_[k]),
                cov=np.atleast_2d(self.covariances_[k]),
                allow_singular=True
            )
        log_prob_norm = logsumexp(log_prob, axis=1, keepdims=True)
        resp = np.exp(log_prob - log_prob_norm)
        return resp

    def sample(self, n_samples):
        component_choices = np.random.choice(self.n_components, size=n_samples, p=self.weights_gmm)
        samples = []
        for k in component_choices:
            samples.append(np.random.multivariate_normal(self.means_[k], self.covariances_[k]))
        return np.vstack(samples)

# Class for MAP-based GMM and its EM training
class MAPGMM:
    def __init__(self, n_components, bm_s, max_iter=100, tol=1e-6, lambda_smooth=50.0, sigma_x=1.0):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_smooth = lambda_smooth
        self.sigma_x = sigma_x
        self.bm_s = bm_s

    def fit(self, X, sample_weights, batch_size=5):
        N, T, D = X.shape
        sample_weights = sample_weights / (np.sum(sample_weights, axis=0, keepdims=True) + 1e-12)
        z_t = np.sum(sample_weights[:,:,None]*X, axis=0)  # (T,D)
        z_history = []
        self.means_ = np.copy(z_t[np.random.choice(T, self.n_components, replace=False)])  # (K,D)
        self.weights_gmm = np.ones(self.n_components) / self.n_components
        global_cov = np.var(X.reshape(-1,D), axis=0) + 1e-3
        self.covariances_ = [np.diag(global_cov) for _ in range(self.n_components)]
        min_var = 1e-3
        min_weight = 1e-8
        for iteration in range(self.max_iter):
            # ---------- E-step----------
            gamma = np.zeros((self.n_components, T))
            for t in range(T):
                for k in range(self.n_components):
                    try:
                        gamma[k,t] = self.weights_gmm[k] * multivariate_normal.pdf(
                            z_t[t], mean=self.means_[k], cov=self.covariances_[k], allow_singular=True
                        )
                    except np.linalg.LinAlgError:
                        gamma[k,t] = 1e-12
                gamma[:,t] /= np.sum(gamma[:,t]) + 1e-12
            z_new = np.zeros_like(z_t)
            for d in range(D):
                if d == 0:
                    z_new[:, d] = z_t[:, d]
                    continue
                main_diag = np.zeros(T)
                off_diag = -self.lambda_smooth * np.ones(T-1)
                b = np.zeros(T)
                for t in range(T):
                    W_t = np.sum(sample_weights[:,t]) / (self.sigma_x**2)
                    y_t = np.sum(sample_weights[:,t,None] * X[:,t,d][:,None], axis=0) / (np.sum(sample_weights[:,t])+1e-12)
                    Sigma_inv = np.sum([gamma[k,t]/(self.covariances_[k][d,d]+1e-12) for k in range(self.n_components)])
                    mu_term = np.sum([gamma[k,t]*self.means_[k,d]/(self.covariances_[k][d,d]+1e-12) for k in range(self.n_components)])
                    main_diag[t] = W_t + Sigma_inv + 2*self.lambda_smooth
                    b[t] = W_t * y_t + mu_term
                main_diag[0] -= self.lambda_smooth
                main_diag[-1] -= self.lambda_smooth
                ab = np.zeros((3,T))
                ab[0,1:] = off_diag
                ab[1,:] = main_diag
                ab[2,:-1] = off_diag
                z_new[:,d] = solve_banded((1,1), ab, b)

            # ---------- M-step----------
            Nk = np.sum(gamma, axis=1) + 1e-12
            self.weights_gmm = np.maximum(Nk / np.sum(Nk), min_weight)
            self.weights_gmm /= np.sum(self.weights_gmm)
            for k in range(self.n_components):
                self.means_[k] = (gamma[k,:,None] * z_new).sum(axis=0) / Nk[k]
            for k in range(self.n_components):
                diff = z_new - self.means_[k]
                cov_diag = (gamma[k,:,None] * (diff**2)).sum(axis=0)/Nk[k]
                cov_diag = np.maximum(cov_diag, min_var)
                self.covariances_[k] = np.diag(cov_diag)
            delta = np.linalg.norm(z_new - z_t)
            z_t = z_new
            z_t_to_save = z_t.copy()
            z_t_to_save[:, 0] = self.bm_s
            z_history.append(z_t_to_save)
            if len(z_history) > batch_size:
                z_history.pop(0)
            if delta < self.tol:
                print(f"Converged at iteration {iteration}")
                break
        self.z_ = z_t
        z_for_gmm = np.concatenate(z_history, axis=0)
        gmm = GaussianMixture(n_components=self.n_components, max_iter=200)
        gmm.fit(z_for_gmm)
        self.gmm_ = gmm
        self.means_ = gmm.means_
        self.covariances_ = gmm.covariances_
        self.weights_gmm = gmm.weights_

# Build the （N,T,D）dataset required for demonstration learning
all_x_train_data_list = []
bm_s_x = None
for i in range(N_demos):
    gmm_dmp_x = GMM_DMP.EmGmmDMP(data_set=x_demos[i], n_rbf=10)
    # (T,) -> (T,1)
    s = gmm_dmp_x.bm_s[:, None]
    f = gmm_dmp_x.bm_f_target[:, None]
    sf = np.concatenate([s, f], axis=1) # (T,2)
    all_x_train_data_list.append(sf)
    if bm_s_x is None:
        bm_s_x = gmm_dmp_x.bm_s
all_x_train_data_array = np.stack(all_x_train_data_list, axis=0)
all_y_train_data_list = []
bm_s_y = None
for i in range(N_demos):
    gmm_dmp_y = GMM_DMP.EmGmmDMP(data_set=y_demos[i], n_rbf=10)
    # (T,) -> (T,1)
    s = gmm_dmp_y.bm_s[:, None]
    f = gmm_dmp_y.bm_f_target[:, None]
    sf = np.concatenate([s, f], axis=1)
    all_y_train_data_list.append(sf)
    if bm_s_y is None:
        bm_s_y = gmm_dmp_y.bm_s
all_y_train_data_array = np.stack(all_y_train_data_list, axis=0)
all_z_train_data_list = []
bm_s_z = None
for i in range(N_demos):
    gmm_dmp_z = GMM_DMP.EmGmmDMP(data_set=z_demos[i], n_rbf=10)
    # (T,) -> (T,1)
    s = gmm_dmp_z.bm_s[:, None]
    f = gmm_dmp_z.bm_f_target[:, None]
    sf = np.concatenate([s, f], axis=1)
    all_z_train_data_list.append(sf)
    if bm_s_z is None:
        bm_s_z = gmm_dmp_z.bm_s
all_z_train_data_array = np.stack(all_z_train_data_list, axis=0)

# Learning from demonstrations
n_rbf = 50
batch_size = 10
max_iter = 100
print('QA-DMP learning...')
x_space2_weights_2d_final = x_space2_final_weights.reshape((N_demos, L_demos))
x_space2_gmm_plus = MAPGMM(n_components=n_rbf, bm_s=bm_s_x, max_iter=max_iter, lambda_smooth=lambda_s)
x_space2_gmm_plus.fit(all_x_train_data_array, x_space2_weights_2d_final, batch_size=batch_size)
y_space2_weights_2d_final = y_space2_final_weights.reshape((N_demos, L_demos))
y_space2_gmm_plus = MAPGMM(n_components=n_rbf, bm_s=bm_s_y, max_iter=max_iter, lambda_smooth=lambda_s)
y_space2_gmm_plus.fit(all_y_train_data_array, y_space2_weights_2d_final, batch_size=batch_size)
z_space2_weights_2d_final = z_space2_final_weights.reshape((N_demos, L_demos))
z_space2_gmm_plus = MAPGMM(n_components=n_rbf, bm_s=bm_s_z, max_iter=max_iter)
z_space2_gmm_plus.fit(all_z_train_data_array, z_space2_weights_2d_final, batch_size=batch_size)

print('DMP learning...')
x_basic_gmm_train_data = []
for i in range(N_demos):
     gmm_dmp_x = GMM_DMP.EmGmmDMP(data_set=x_demos[i], n_rbf=10)
     x_train_data = gmm_dmp_x.train_data
     x_basic_gmm_train_data.append(x_train_data)
all_x_train_data = np.vstack(x_basic_gmm_train_data)
x_space4_gmm = VanillaGMM(n_components=n_rbf, max_iter=max_iter)
x_space4_gmm.fit(all_x_train_data)
y_basic_gmm_train_data = []
for i in range(N_demos):
     gmm_dmp_y = GMM_DMP.EmGmmDMP(data_set=y_demos[i], n_rbf=10)
     y_train_data = gmm_dmp_y.train_data
     y_basic_gmm_train_data.append(y_train_data)
all_y_train_data = np.vstack(y_basic_gmm_train_data)
y_space4_gmm = VanillaGMM(n_components=n_rbf, max_iter=max_iter)
y_space4_gmm.fit(all_y_train_data)
z_basic_gmm_train_data = []
for i in range(N_demos):
     gmm_dmp_z = GMM_DMP.EmGmmDMP(data_set=z_demos[i], n_rbf=10)
     z_train_data = gmm_dmp_z.train_data
     z_basic_gmm_train_data.append(z_train_data)
all_z_train_data = np.vstack(z_basic_gmm_train_data) # 所有演示对应训练集打包成一个大训练集
z_space4_gmm = VanillaGMM(n_components=n_rbf, max_iter=max_iter)
z_space4_gmm.fit(all_z_train_data)

# Generating outputs
print('Generating...')
start_x = baseline_x[0]
end_x = baseline_x[-1]
start_y = baseline_y[0]
end_y = baseline_y[-1]
start_z = baseline_z[0]
end_z = baseline_z[-1]
x_space2_sampled_f = []
x_space2_outputs = []
x_space4_sampled_f = []
x_space4_outputs = []
y_space2_sampled_f = []
y_space2_outputs = []
y_space4_sampled_f = []
y_space4_outputs = []
z_space2_sampled_f = []
z_space2_outputs = []
z_space4_sampled_f = []
z_space4_outputs = []
for i in range(cnt_sample):
    space2_x_repo, f = weighted_gmm_reproduction(weighted_gmm=x_space2_gmm_plus, bm_s=bm_s_x, start=start_x, target=end_x)
    x_space2_sampled_f.append(f)
    x_space2_outputs.append(space2_x_repo)
    space2_y_repo, f = weighted_gmm_reproduction(weighted_gmm=y_space2_gmm_plus, bm_s=bm_s_y, start=start_y, target=end_y)
    y_space2_sampled_f.append(f)
    y_space2_outputs.append(space2_y_repo)
    space2_z_repo, f = weighted_gmm_reproduction(weighted_gmm=z_space2_gmm_plus, bm_s=bm_s_z, start=start_z, target=end_z)
    z_space2_sampled_f.append(f)
    z_space2_outputs.append(space2_z_repo)

    space4_x_repo, f = weighted_gmm_reproduction(weighted_gmm=x_space4_gmm, bm_s=bm_s_x, start=start_x, target=end_x)
    x_space4_sampled_f.append(f)
    x_space4_outputs.append(space4_x_repo)
    space4_y_repo, f = weighted_gmm_reproduction(weighted_gmm=y_space4_gmm, bm_s=bm_s_y, start=start_y, target=end_y)
    y_space4_sampled_f.append(f)
    y_space4_outputs.append(space4_y_repo)
    space4_z_repo, f = weighted_gmm_reproduction(weighted_gmm=z_space4_gmm, bm_s=bm_s_z, start=start_z, target=end_z)
    z_space4_sampled_f.append(f)
    z_space4_outputs.append(space4_z_repo)

# Visualization: The output result of DMP
fig = plt.figure(figsize=(7.5, 6))
ax = fig.add_subplot(projection='3d')
for i in range(N_demos):
    if i == 0:
        ax.plot(x_demos[i], y_demos[i], z_demos[i], 'b--', label='Demos')
    else:
        ax.plot(x_demos[i], y_demos[i], z_demos[i], 'b--', label=None)
for i in range(cnt_sample):
    if i == 0:
        ax.plot(x_space4_outputs[i], y_space4_outputs[i], z_space4_outputs[i], 'g-', label='Outputs')
    else:
        ax.plot(x_space4_outputs[i], y_space4_outputs[i], z_space4_outputs[i], 'g-')
ax.plot(baseline_x, baseline_y, baseline_z, 'r--', label='Desired demo')
ax.set_xlabel('x', labelpad=0)
ax.set_ylabel('y', labelpad=0)
ax.set_zlabel('z', labelpad=0)
ax.view_init(elev=28, azim=-71)
plt.legend(frameon=False, loc='best')
plt.tight_layout()
plt.show()

# Visualization: The output result of QA-DMP
fig = plt.figure(figsize=(7.5, 6))
ax = fig.add_subplot(projection='3d')
for i in range(N_demos):
    if i == 0:
        ax.plot(x_demos[i], y_demos[i], z_demos[i], 'b--', label='Demos')
    else:
        ax.plot(x_demos[i], y_demos[i], z_demos[i], 'b--', label=None)
for i in range(cnt_sample):
    if i == 0:
        ax.plot(x_space2_outputs[i], y_space2_outputs[i], z_space2_outputs[i], 'g-', label='Outputs')
    else:
        ax.plot(x_space2_outputs[i], y_space2_outputs[i], z_space2_outputs[i], 'g-')
ax.plot(baseline_x, baseline_y, baseline_z, 'r--', label='Desired demo')
ax.set_xlabel('x', labelpad=0)
ax.set_ylabel('y', labelpad=0)
ax.set_zlabel('z', labelpad=0)
ax.view_init(elev=28, azim=-71)
plt.legend(frameon=False, loc='best')
plt.tight_layout()
plt.show()

# Metrics calculation: Compactness, smoothness, and similarity to the expected presentation
def compute_compactness_3d(x_list, y_list, z_list):
    N = len(x_list)
    pairwise_dists = []

    for i, j in combinations(range(N), 2):
        xi, yi, zi = x_list[i], y_list[i], z_list[i]
        xj, yj, zj = x_list[j], y_list[j], z_list[j]
        dist = np.mean(np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2))
        pairwise_dists.append(dist)

    return np.mean(pairwise_dists)

def compute_average_jerk_3d(x_list, y_list, z_list):
    jerks = []
    for x, y, z in zip(x_list, y_list, z_list):
        for traj in [x, y, z]:
            vel = np.gradient(traj)
            acc = np.gradient(vel)
            jerk = np.gradient(acc)
            jerks.append(np.mean(np.abs(jerk)))
    return np.mean(jerks)

def compute_average_distance_to_baseline_3d(x_list, y_list, z_list,
                                            baseline_x, baseline_y, baseline_z):
    dists = []
    for x, y, z in zip(x_list, y_list, z_list):
        dist = np.sqrt((x - baseline_x)**2 +
                       (y - baseline_y)**2 +
                       (z - baseline_z)**2)
        dists.append(np.mean(dist))
    return np.mean(dists)

compactness_space2 = compute_compactness_3d(x_space2_outputs, y_space2_outputs, z_space2_outputs)
jerk_space2 = compute_average_jerk_3d(x_space2_outputs, y_space2_outputs, z_space2_outputs)
dist_to_baseline_space2 = compute_average_distance_to_baseline_3d(x_space2_outputs, y_space2_outputs, z_space2_outputs, baseline_x, baseline_y, baseline_z)

compactness_space4 = compute_compactness_3d(x_space4_outputs, y_space4_outputs, z_space4_outputs)
jerk_space4 = compute_average_jerk_3d(x_space4_outputs, y_space4_outputs, z_space4_outputs)
dist_to_baseline_space4 = compute_average_distance_to_baseline_3d(x_space4_outputs, y_space4_outputs, z_space4_outputs, baseline_x, baseline_y, baseline_z)

print("DMP: compactness={}, jerk={}, distance to baseline={}".format(compactness_space4, jerk_space4, dist_to_baseline_space4))
print("QA-DMP: compactness={}, jerk={}, distance to baseline={}".format(compactness_space2, jerk_space2, dist_to_baseline_space2))

