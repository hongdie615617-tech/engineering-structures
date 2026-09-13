import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import xlsxwriter
import xlrd
import matplotlib.pylab as plt




def load_target(filename='Predicted_result.xlsx'):
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_name('sheet1')
    m = sheet.nrows

    h = np.zeros(m)
    u = np.zeros(m)

    for i in range(m):
        h[i] = sheet.cell(i, 0).value
        u[i] = sheet.cell(i, 1).value

    # Sort by h to make the deformation path ordered
    idx = np.argsort(h)
    h = h[idx]
    u = u[idx]

    return h, u


def save_design_result(h, target_u, U_ad, theta_ad, R_ad, filename):
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet('AD_only_result')

    headers = [
        'h',
        'target_U',
        'U_AD_only_random_single_start',
        'theta_AD_only_random_single_start',
        'R_E',
        'R_E_squared'
    ]

    for col, header in enumerate(headers):
        worksheet.write(0, col, header)

    for i in range(len(h)):
        worksheet.write(i + 1, 0, float(h[i]))
        worksheet.write(i + 1, 1, float(target_u[i]))
        worksheet.write(i + 1, 2, float(U_ad[i]))
        worksheet.write(i + 1, 3, float(theta_ad[i]))
        worksheet.write(i + 1, 4, float(R_ad[i]))
        worksheet.write(i + 1, 5, float(R_ad[i] ** 2))

    workbook.close()
    print(f'\nResult saved to: {filename}')


def save_training_log(log_data, filename):
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet('training_log')

    headers = [
        'Epoch',
        'loss_D',
        'loss_data',
        'loss_G',
        'loss_smooth',
        'a',
        'alpha_rad',
        'alpha_degree'
    ]

    for col, header in enumerate(headers):
        worksheet.write(0, col, header)

    for row, item in enumerate(log_data, start=1):
        worksheet.write(row, 0, item['epoch'])
        worksheet.write(row, 1, item['loss_D'])
        worksheet.write(row, 2, item['loss_data'])
        worksheet.write(row, 3, item['loss_G'])
        worksheet.write(row, 4, item['loss_smooth'])
        worksheet.write(row, 5, item['a'])
        worksheet.write(row, 6, item['alpha_rad'])
        worksheet.write(row, 7, item['alpha_deg'])

    workbook.close()
    print(f'Training log saved to: {filename}')


def save_summary(summary, filename):
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet('summary')

    headers = [
        'Method',
        'loss_D',
        'loss_data',
        'loss_G',
        'R_E_max',
        'a',
        'alpha_rad',
        'alpha_degree',
        'n',
        'epochs',
        'learning_rate',
        'lambda_G',
        'lambda_smooth',
        'elapsed_time_s',
        'delta_alpha_init',
        'delta_a_init',
        'theta_init_first',
        'theta_init_last',
        'theta_noise_std',
        'use_sort'
    ]

    for col, header in enumerate(headers):
        worksheet.write(0, col, header)

    values = [
        summary['method'],
        summary['loss_D'],
        summary['loss_data'],
        summary['loss_G'],
        summary['R_E_max'],
        summary['a'],
        summary['alpha_rad'],
        summary['alpha_deg'],
        summary['n'],
        summary['epochs'],
        summary['learning_rate'],
        summary['lambda_G'],
        summary['lambda_smooth'],
        summary['elapsed_time_s'],
        summary['delta_alpha_init'],
        summary['delta_a_init'],
        summary['theta_init_first'],
        summary['theta_init_last'],
        summary['theta_noise_std'],
        int(summary['use_sort'])
    ]

    for col, value in enumerate(values):
        worksheet.write(1, col, value)

    workbook.close()
    print(f'Summary saved to: {filename}')


def run_random_single_start_ad_only(
    h_list,
    ut,
    n_value=6.0,
    epochs=10000,
    learning_rate=1e-2,
    lambda_G=1.0,
    lambda_smooth=0.0,
    use_sort=True,
    log_every=50,
    theta_noise_std=0.25
):
    """
    Random single-start AD-only inverse design.

    No neural network is used.

    Trainable variables:
        delta_alpha
        delta_a
        theta_raw

    Main loss:
        loss_D = loss_data + lambda_G * loss_G + lambda_smooth * loss_smooth

    For fair comparison with the PINN loss:
        lambda_G = 1.0
        lambda_smooth = 0.0

    Randomness:
        No fixed seed is used.
        Each run randomly initializes:
            delta_alpha
            delta_a
            theta_init
    """

    tf.reset_default_graph()

    # No fixed seed:
    # Do not call tf.set_random_seed(...)
    # Do not use np.random.RandomState(fixed_seed)
    rng = np.random.default_rng()

    h_number = h_list.shape[0]

    h_np = h_list.reshape(1, -1)
    ut_np = ut.reshape(1, -1)

    h = tf.constant(h_np, tf.float64)
    ut_tf = tf.constant(ut_np, tf.float64)

    # ============================================================
    # 1. Random initialization of trainable geometry variables
    # ============================================================

    delta_alpha_init = rng.normal(loc=0.0, scale=0.5, size=(1, 1))
    delta_a_init = rng.normal(loc=0.0, scale=0.5, size=(1, 1))

    delta_alpha = tf.Variable(delta_alpha_init, dtype=tf.float64, name='delta_alpha')
    delta_a = tf.Variable(delta_a_init, dtype=tf.float64, name='delta_a')

    n = tf.constant([[n_value]], tf.float64)
    r = tf.constant([[1.0]], tf.float64)
    ka = tf.constant([[1.0]], tf.float64)

    c = 2.0 * r * tf.sin(np.pi / n)

    # Bounded alpha:
    #     0 < alpha < pi
    alpha_min = 1e-6
    alpha_max = np.pi - 1e-6
    alpha = alpha_min + (alpha_max - alpha_min) * tf.nn.sigmoid(delta_alpha)

    # Bounded a:
    #     0 < a < c / sin(alpha)
    a = tf.nn.sigmoid(delta_a) * c / tf.sin(alpha)

    asin_arg = a * tf.sin(alpha) / c
    asin_arg = tf.clip_by_value(asin_arg, -1.0 + 1e-12, 1.0 - 1e-12)

    b = c / tf.sin(alpha) * tf.sin(alpha + tf.asin(asin_arg))

    kb = a / b * ka
    eta = kb / ka

    # ============================================================
    # 2. Random initialization of direct trainable theta path
    # ============================================================

    # Base monotonic path
    theta_base = np.linspace(0.9 * np.pi, -0.9 * np.pi, h_number)

    # Random perturbation
    theta_noise = rng.normal(loc=0.0, scale=theta_noise_std, size=h_number)

    theta_init = theta_base + theta_noise
    theta_init = np.clip(theta_init, -0.95 * np.pi, 0.95 * np.pi)

    # Convert theta_init to theta_raw_init because:
    #     theta = pi * sin(theta_raw)
    theta_raw_init = np.arcsin(np.clip(theta_init / np.pi, -0.999999, 0.999999))
    theta_raw_init = theta_raw_init.reshape(1, -1)

    theta_raw = tf.Variable(theta_raw_init, dtype=tf.float64, name='theta_raw')

    theta = np.pi * tf.sin(theta_raw)

    # Same sorting operation as the PINN code.
    # It gives a branch-ordering constraint but does not introduce NN weights.
    if use_sort:
        theta = tf.reshape(theta, [h_number])
        theta = tf.sort(theta, direction='DESCENDING')
        theta = tf.reshape(theta, [1, h_number])

    # ============================================================
    # 3. Physics model
    # ============================================================

    ad = tf.sqrt(
        tf.square(h / 2.0)
        + 4.0 * r ** 2 * tf.sin(theta / 2.0) ** 2
    )

    bd = tf.sqrt(
        tf.square(h / 2.0)
        + 4.0 * r ** 2 * tf.sin(theta / 2.0 + np.pi / n) ** 2
    )

    eps = tf.constant(1e-12, tf.float64)
    ad_safe = tf.maximum(ad, eps)
    bd_safe = tf.maximum(bd, eps)

    U = n * tf.square(ad - a) + n * eta * tf.square(bd - b)

    R_E = (
        (1.0 - a / ad_safe) * tf.sin(theta)
        + eta * (1.0 - b / bd_safe) * tf.sin(theta + 2.0 * np.pi / n)
    )

    # ============================================================
    # 4. Losses
    # ============================================================

    loss_data = tf.reduce_mean(tf.square(U - ut_tf))
    loss_G = tf.reduce_mean(tf.square(R_E))

    # Optional diagnostic/regularization term.
    # Keep lambda_smooth = 0.0 for fair comparison if PINN does not use this term.
    if h_number >= 3:
        dtheta_1 = theta[:, 1:] - theta[:, :-1]
        dtheta_2 = dtheta_1[:, 1:] - dtheta_1[:, :-1]
        loss_smooth = tf.reduce_mean(tf.square(dtheta_2))
    else:
        loss_smooth = tf.constant(0.0, tf.float64)

    loss_D = loss_data + lambda_G * loss_G + lambda_smooth * loss_smooth

    # ============================================================
    # 5. Optimizer
    # ============================================================

    train_vars = [delta_alpha, delta_a, theta_raw]

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss_D,
        var_list=train_vars
    )

    # ============================================================
    # 6. Training
    # ============================================================

    best = {
        'loss_D': np.inf,
        'loss_data': np.inf,
        'loss_G': np.inf,
        'loss_smooth': np.inf,
        'a': None,
        'alpha': None,
        'U': None,
        'theta': None,
        'R_E': None
    }

    log_data = []

    print('\nInitial settings')
    print('----------------')
    print(f'delta_alpha_init = {delta_alpha_init[0, 0]:.12e}')
    print(f'delta_a_init     = {delta_a_init[0, 0]:.12e}')
    print(f'theta_init[0]    = {theta_init[0]:.12e}')
    print(f'theta_init[-1]   = {theta_init[-1]:.12e}')
    print(f'theta_noise_std  = {theta_noise_std:.12e}')
    print(f'use_sort         = {use_sort}')

    start_time = time.time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Initial physical variables after mapping
        a_initial, alpha_initial, loss_initial, loss_data_initial, loss_G_initial = sess.run(
            [a, alpha, loss_D, loss_data, loss_G]
        )

        print('\nInitial physical variables')
        print('--------------------------')
        print(f'a_initial         = {a_initial[0, 0]:.12e}')
        print(f'alpha_initial_rad = {alpha_initial[0, 0]:.12e}')
        print(f'alpha_initial_deg = {alpha_initial[0, 0] / np.pi * 180.0:.12e}')
        print(f'initial loss_D    = {loss_initial:.12e}')
        print(f'initial loss_data = {loss_data_initial:.12e}')
        print(f'initial loss_G    = {loss_G_initial:.12e}')

        print('\nStart random single-start AD-only training')
        print('------------------------------------------')
        print('Epoch | loss_D | loss_data | loss_G | a | alpha_deg')

        for epoch in range(epochs):
            _, loss_D_v, loss_data_v, loss_G_v, loss_smooth_v, a_v, alpha_v = sess.run(
                [optimizer, loss_D, loss_data, loss_G, loss_smooth, a, alpha]
            )

            if loss_D_v < best['loss_D']:
                U_v, theta_v, R_v = sess.run([U, theta, R_E])

                best['loss_D'] = loss_D_v
                best['loss_data'] = loss_data_v
                best['loss_G'] = loss_G_v
                best['loss_smooth'] = loss_smooth_v
                best['a'] = a_v
                best['alpha'] = alpha_v
                best['U'] = U_v
                best['theta'] = theta_v
                best['R_E'] = R_v

            if epoch % log_every == 0 or epoch == epochs - 1:
                log_data.append({
                    'epoch': epoch,
                    'loss_D': float(loss_D_v),
                    'loss_data': float(loss_data_v),
                    'loss_G': float(loss_G_v),
                    'loss_smooth': float(loss_smooth_v),
                    'a': float(a_v[0, 0]),
                    'alpha_rad': float(alpha_v[0, 0]),
                    'alpha_deg': float(alpha_v[0, 0] / np.pi * 180.0)
                })

            if epoch % 500 == 0 or epoch == epochs - 1:
                print(
                    f'{epoch:05d} | '
                    f'{loss_D_v:.6e} | '
                    f'{loss_data_v:.6e} | '
                    f'{loss_G_v:.6e} | '
                    f'{a_v[0, 0]:.6f} | '
                    f'{alpha_v[0, 0] / np.pi * 180.0:.6f}'
                )

    elapsed_time = time.time() - start_time

    best['elapsed_time'] = elapsed_time
    best['delta_alpha_init'] = float(delta_alpha_init[0, 0])
    best['delta_a_init'] = float(delta_a_init[0, 0])
    best['theta_init_first'] = float(theta_init[0])
    best['theta_init_last'] = float(theta_init[-1])
    best['theta_noise_std'] = float(theta_noise_std)

    return best, log_data


# ============================================================
# Main program
# ============================================================

if __name__ == '__main__':

    # ---------------- User settings ----------------

    target_filename = 'Predicted_result.xlsx'

    # Fixed integer side number.
    # n is not optimized by AD because it is an integer variable.
    n_value = 6.0

    epochs = 10000
    learning_rate = 1e-2

    # Same main loss as PINN:
    #     loss_D = loss_data + loss_G
    lambda_G = 1.0

    # Keep zero for fair comparison if PINN does not use smoothing loss.
    lambda_smooth = 0.0

    # Same sorting operation as the PINN code.
    use_sort = True

    log_every = 50

    # Random theta perturbation level.
    # Increase this if you want to test stronger initialization sensitivity.
    theta_noise_std = 0.25

    # ---------------- Load target ----------------

    h_list, ut = load_target(target_filename)

    print('\nTarget loaded')
    print('-------------')
    print(f'File      : {target_filename}')
    print(f'Points    : {len(h_list)}')
    print(f'h_min     : {np.min(h_list):.12e}')
    print(f'h_max     : {np.max(h_list):.12e}')
    print(f'U_min     : {np.min(ut):.12e}')
    print(f'U_max     : {np.max(ut):.12e}')

    # ---------------- Random single-start AD-only inverse design ----------------

    best, log_data = run_random_single_start_ad_only(
        h_list=h_list,
        ut=ut,
        n_value=n_value,
        epochs=epochs,
        learning_rate=learning_rate,
        lambda_G=lambda_G,
        lambda_smooth=lambda_smooth,
        use_sort=use_sort,
        log_every=log_every,
        theta_noise_std=theta_noise_std
    )

    # ---------------- Final result ----------------

    U_best = best['U'].reshape(-1)
    theta_best = best['theta'].reshape(-1)
    R_best = best['R_E'].reshape(-1)

    L_G_best = np.mean(R_best ** 2)
    Rmax_best = np.max(np.abs(R_best))

    print('\n\n============================================')
    print('Random single-start AD-only inverse design result')
    print('============================================')
    print(f"loss_D       = {best['loss_D']:.12e}")
    print(f"loss_data    = {best['loss_data']:.12e}")
    print(f"loss_G       = {best['loss_G']:.12e}")
    print(f"L_G check    = {L_G_best:.12e}")
    print(f"R_E max      = {Rmax_best:.12e}")
    print(f"a            = {best['a'][0, 0]:.12e}")
    print(f"alpha_rad    = {best['alpha'][0, 0]:.12e}")
    print(f"alpha_deg    = {best['alpha'][0, 0] / np.pi * 180.0:.12e}")
    print(f"n            = {n_value}")
    print(f"elapsed time = {best['elapsed_time']:.6f} s")

    # ---------------- Save files ----------------

    save_design_result(
        h=h_list,
        target_u=ut,
        U_ad=U_best,
        theta_ad=theta_best,
        R_ad=R_best,
        filename='AD_only_random_single_start_design_result.xlsx'
    )

    save_training_log(
        log_data=log_data,
        filename='AD_only_random_single_start_training_process.xlsx'
    )

    summary = {
        'method': 'AD_only_random_single_start',
        'loss_D': float(best['loss_D']),
        'loss_data': float(best['loss_data']),
        'loss_G': float(best['loss_G']),
        'R_E_max': float(Rmax_best),
        'a': float(best['a'][0, 0]),
        'alpha_rad': float(best['alpha'][0, 0]),
        'alpha_deg': float(best['alpha'][0, 0] / np.pi * 180.0),
        'n': float(n_value),
        'epochs': int(epochs),
        'learning_rate': float(learning_rate),
        'lambda_G': float(lambda_G),
        'lambda_smooth': float(lambda_smooth),
        'elapsed_time_s': float(best['elapsed_time']),
        'delta_alpha_init': float(best['delta_alpha_init']),
        'delta_a_init': float(best['delta_a_init']),
        'theta_init_first': float(best['theta_init_first']),
        'theta_init_last': float(best['theta_init_last']),
        'theta_noise_std': float(best['theta_noise_std']),
        'use_sort': bool(use_sort)
    }

    save_summary(
        summary=summary,
        filename='AD_only_random_single_start_summary.xlsx'
    )

    # ---------------- Plot energy curve ----------------

    plt.figure()
    plt.plot(h_list.reshape(-1), U_best.reshape(-1), label='AD-only random single-start design')
    plt.plot(h_list.reshape(-1), ut.reshape(-1), 'o', label='Target')
    plt.xlabel('h')
    plt.ylabel('U')
    plt.ylim(0.0, max(0.03, 1.2 * np.max(ut)))
    plt.legend()
    plt.title('Random single-start AD-only inverse design: energy curve')
    plt.show()

    # ---------------- Plot theta path ----------------

    plt.figure()
    plt.plot(h_list.reshape(-1), theta_best.reshape(-1), label='AD-only random single-start theta')
    plt.xlabel('h')
    plt.ylabel('theta')
    plt.legend()
    plt.title('Random single-start AD-only inverse design: rotational path')
    plt.show()

    # ---------------- Plot residual ----------------

    plt.figure()
    plt.semilogy(h_list.reshape(-1), np.abs(R_best.reshape(-1)), label='|R_E|')
    plt.xlabel('h')
    plt.ylabel('|R_E|')
    plt.legend()
    plt.title('Random single-start AD-only inverse design: residual')
    plt.show()