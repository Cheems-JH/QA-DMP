###########################################################
# Original GMM-based DMP
###########################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture

class EmGmmDMP:
    def __init__(self, data_set, n_rbf=100, alpha_y=60, beta_y=60/4.0, alpha_x=1.0, cs_runtime=1.0):
        self.y_demo = data_set
        self.n_rbf = n_rbf
        self.data_len = len(data_set)
        self.y0 = data_set[0]
        self.g = data_set[-1]
        self.alpha_y = alpha_y
        self.beta_y = beta_y
        self.alpha_x = alpha_x
        self.cs_runtime = cs_runtime
        self.x0 = 1.0
        self.tau_t = 1.0
        self.start = 0.0
        self.target = 0.0
        self.tau = 0.0
        self.x_auxiliary = self.x0
        self.y = self.y0
        self.dy = 0.0
        self.ddy = 0.0
        self.dt = self.cs_runtime / self.data_len
        self.time_steps = round(self.cs_runtime / self.dt)
        self.train_data = []
        self.gmm, self.bm_s, self.bm_f_target, self.x_track = self.training()

    # training with demo
    def training(self):
        x_track = np.zeros(self.time_steps)
        tmp_x = self.x0
        for i in range(self.time_steps):
            x_track[i] = tmp_x
            dx = - self.alpha_x * tmp_x * self.dt
            tmp_x = tmp_x + self.tau_t * dx
        x = np.linspace(0, self.cs_runtime, len(self.y_demo))
        y = np.zeros(self.time_steps)
        y_tmp = interp1d(x, self.y_demo)
        for k in range(self.time_steps):
            y[k] = y_tmp(k * self.dt)
        dy_demo = np.gradient(y) / self.dt
        ddy_demo = np.gradient(dy_demo) / self.dt
        bm_f_target = ddy_demo - self.alpha_y * (self.beta_y * (self.g - self.y_demo) - dy_demo)
        bm_s = np.zeros(len(self.y_demo))
        for i in range(len(bm_s)):
            bm_s[i] = x_track[i] * (self.g - self.y0)

        train_data = np.vstack((bm_s, bm_f_target)).T
        self.train_data = train_data

        gmm = GaussianMixture(n_components=self.n_rbf, covariance_type='full', init_params='random')
        gmm.fit(train_data)

        return gmm, bm_s, bm_f_target, x_track

    # reproduction based on DMP
    def reproduction(self, start=None, target=None, tau=None):
        # set temporal scaling
        if tau is None:
            time_steps = self.time_steps
        else:
            time_steps = round(self.time_steps / tau)
        # set parameters
        if start is None:
            self.start = self.y0
        else:
            self.start = start
        if target is None:
            self.target = self.g
        else:
            self.target = target
        if tau is None:
            self.tau = self.tau_t
        else:
            self.tau = tau

        predictions = self.gmm.predict(self.train_data)
        samples_input = []
        samples_output = []
        for k in range(len(predictions)):
            x_fixed = self.train_data[k][0]
            component = predictions[k]
            mean = self.gmm.means_[component]
            cov = self.gmm.covariances_[component]
            mu_x = mean[0]
            mu_y = mean[1]
            sigma_xx = cov[0, 0]
            sigma_yy = cov[1, 1]
            sigma_xy = cov[0, 1]
            y_cond_mean = mu_y + sigma_xy / sigma_xx * (x_fixed - mu_x)
            y_cond_var = sigma_yy - (sigma_xy ** 2) / sigma_xx
            y_cond_std = np.sqrt(y_cond_var)
            y_sample = np.random.normal(y_cond_mean, y_cond_std)
            samples_input.append(x_fixed)
            samples_output.append(y_sample)

        bm_s_generate = samples_input
        bm_f_target_generate = samples_output
        x_g = bm_s_generate / (self.g - self.y0)
        t_g = np.zeros_like(x_g)
        for k in range(len(bm_s_generate) - 1):
            delta_k = (x_g[k + 1] - x_g[k]) / self.tau
            t_g[k] = delta_k / (- self.alpha_x * x_g[k])
        y_reproduce = np.zeros(len(bm_s_generate))
        dy_reproduce = np.zeros(len(bm_s_generate))
        ddy_reproduce = np.zeros(len(bm_s_generate))
        tmp_y = self.start
        tmp_dy = 0.0
        tmp_ddy = 0.0
        gain = (self.target - self.start) / (self.g - self.y0)
        for k in range(len(bm_s_generate)):
            y_reproduce[k] = tmp_y
            dy_reproduce[k] = tmp_dy
            ddy_reproduce[k] = tmp_ddy
            tmp_ddy = self.alpha_y * (self.beta_y * (self.target - tmp_y) - tmp_dy) + bm_f_target_generate[k] * gain
            tmp_dy += self.tau * tmp_ddy * t_g[k]
            tmp_y += self.tau * tmp_dy * t_g[k]
        return y_reproduce, samples_input, samples_output


# ------ Test with a simple sin 1-D trajectory
if __name__ == "__main__":
    data_len = 1000
    t = np.linspace(0, 1.5 * np.pi, data_len)
    demo_y = np.sin(t)
    dmp = EmGmmDMP(data_set=demo_y)
    y_g, _, _ = dmp.reproduction()
    plt.figure()
    plt.plot(t, demo_y, 'b', label='demo')
    plt.plot(t, y_g, 'g', label='gmm-dmp repo')
    plt.legend()
    plt.show()

