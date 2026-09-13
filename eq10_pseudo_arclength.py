from __future__ import annotations
import argparse
import csv
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np


class ChiralEq10:
    def __init__(self, a0_bar: float, phi0: float, n: int, energy_factor: float = 1.0):
        self.a0_bar = float(a0_bar)
        self.phi0 = float(phi0)
        self.n = int(n)
        self.energy_factor = float(energy_factor)
        self.feval_count = 0
        self.c_bar = self.compute_c_bar(self.n)
        self.b0_bar = self.compute_b0_bar(self.a0_bar, self.phi0, self.n)
        self.xi = self.a0_bar / self.b0_bar

    @staticmethod
    def compute_c_bar(n: int) -> float:
        return 2.0 * math.sin(math.pi / n)

    @classmethod
    def compute_b0_bar(cls, a0_bar: float, phi0: float, n: int) -> float:
        c_bar = cls.compute_c_bar(n)
        arg = a0_bar * math.sin(phi0) / c_bar
        if abs(arg) >= 1.0:
            raise ValueError("Invalid geometry: asin argument is outside [-1, 1].")
        if abs(math.sin(phi0)) < 1e-12:
            raise ValueError("Invalid geometry: sin(phi0) is too small.")
        return c_bar * math.sin(phi0 + math.asin(arg)) / math.sin(phi0)

    @staticmethod
    def deformed_lengths(h_bar, theta, n: int):
        h_bar = np.asarray(h_bar, dtype=float)
        theta = np.asarray(theta, dtype=float)
        a_d_bar = np.sqrt((h_bar / 2.0) ** 2 + 4.0 * np.sin(theta / 2.0) ** 2)
        b_d_bar = np.sqrt((h_bar / 2.0) ** 2 + 4.0 * np.sin(theta / 2.0 + math.pi / n) ** 2)
        return a_d_bar, b_d_bar

    def energy(self, h_bar, theta):
        a_d_bar, b_d_bar = self.deformed_lengths(h_bar, theta, self.n)
        return self.energy_factor * self.n * (
            (a_d_bar - self.a0_bar) ** 2 + self.xi * (b_d_bar - self.b0_bar) ** 2
        )

    def residual_derivatives(self, h_bar, theta):
        theta_arr = np.asarray(theta, dtype=float)
        self.feval_count += theta_arr.size
        delta = 2.0 * math.pi / self.n
        a_d_bar, b_d_bar = self.deformed_lengths(h_bar, theta_arr, self.n)
        eps_val = 1e-14
        a_d_bar = np.maximum(a_d_bar, eps_val)
        b_d_bar = np.maximum(b_d_bar, eps_val)

        r = (
            (1.0 - self.a0_bar / a_d_bar) * np.sin(theta_arr)
            + self.xi * (1.0 - self.b0_bar / b_d_bar) * np.sin(theta_arr + delta)
        )
        r_theta = (
            self.a0_bar * np.sin(theta_arr) ** 2 / a_d_bar**3
            + (1.0 - self.a0_bar / a_d_bar) * np.cos(theta_arr)
            + self.xi
            * (
                self.b0_bar * np.sin(theta_arr + delta) ** 2 / b_d_bar**3
                + (1.0 - self.b0_bar / b_d_bar) * np.cos(theta_arr + delta)
            )
        )
        r_h = (
            self.a0_bar * h_bar * np.sin(theta_arr) / (4.0 * a_d_bar**3)
            + self.xi * self.b0_bar * h_bar * np.sin(theta_arr + delta) / (4.0 * b_d_bar**3)
        )
        if np.ndim(theta) == 0:
            return float(r), float(r_h), float(r_theta)
        return r, r_h, r_theta

    def residual(self, h_bar, theta):
        r, _, _ = self.residual_derivatives(h_bar, theta)
        return r

    def second_derivative_energy(self, h_bar, theta, dtheta: float = 1e-5):
        theta_p = min(theta + dtheta, math.pi)
        theta_m = max(theta - dtheta, -math.pi)
        if abs(theta_p - theta_m) < 1e-14:
            return math.nan
        u0 = self.energy(h_bar, theta)
        up = self.energy(h_bar, theta_p)
        um = self.energy(h_bar, theta_m)
        return float((up - 2.0 * u0 + um) / (dtheta**2))


@dataclass
class Branch:
    h: np.ndarray
    theta: np.ndarray
    u: np.ndarray
    r: np.ndarray
    d2u: np.ndarray
    is_stable: np.ndarray
    newton_iters: np.ndarray


class ContinuationState:
    def __init__(self):
        self.total_newton_iter = 0


def unique_roots(roots_in, tol: float):
    roots = np.asarray(roots_in, dtype=float)
    roots = roots[np.isfinite(roots)]
    roots.sort()
    if roots.size == 0:
        return np.array([], dtype=float)
    roots_unique = [float(roots[0])]
    for root in roots[1:]:
        if abs(root - roots_unique[-1]) > tol:
            roots_unique.append(float(root))
        else:
            roots_unique[-1] = 0.5 * (roots_unique[-1] + float(root))
    return np.asarray(roots_unique, dtype=float)


def damped_newton_fixed_h(
    model: ChiralEq10,
    theta0: float,
    h_bar: float,
    theta_min: float,
    theta_max: float,
    tol: float,
    max_iter: int,
):
    theta = min(max(float(theta0), theta_min), theta_max)
    iter_count = 0
    for iter_count in range(1, max_iter + 1):
        r, _, r_theta = model.residual_derivatives(h_bar, theta)
        if abs(r) < tol:
            return theta, True, iter_count
        if (not np.isfinite(r)) or (not np.isfinite(r_theta)) or abs(r_theta) < 1e-14:
            return theta, False, iter_count

        step = -r / r_theta
        alpha = 1.0
        r_old = abs(r)
        accepted = False
        for _ls in range(25):
            theta_trial = theta + alpha * step
            if theta_trial < theta_min or theta_trial > theta_max:
                alpha *= 0.5
                continue
            r_trial = abs(model.residual(h_bar, theta_trial))
            if np.isfinite(r_trial) and r_trial < r_old:
                theta = theta_trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            return theta, False, iter_count

        if abs(alpha * step) < tol:
            r_final = abs(model.residual(h_bar, theta))
            return theta, r_final < 1e-8, iter_count

    return theta, abs(model.residual(h_bar, theta)) < 1e-8, iter_count


def compute_tangent(model: ChiralEq10, h_bar: float, theta: float):
    _, r_h, r_theta = model.residual_derivatives(h_bar, theta)
    tangent = np.array([r_theta, -r_h], dtype=float)
    norm_t = float(np.linalg.norm(tangent))
    if norm_t < 1e-14:
        raise RuntimeError("Tangent vector is nearly zero.")
    return tangent / norm_t


def pseudo_arc_corrector(
    model: ChiralEq10,
    x_pred: np.ndarray,
    tangent: np.ndarray,
    theta_min: float,
    theta_max: float,
    tol: float,
    max_iter: int,
):
    x = np.array(x_pred, dtype=float)
    iter_count = 0
    for iter_count in range(1, max_iter + 1):
        h, theta = float(x[0]), float(x[1])
        r, r_h, r_theta = model.residual_derivatives(h, theta)
        g = np.array([r, np.dot(x - x_pred, tangent)], dtype=float)
        if float(np.linalg.norm(g)) < tol:
            return x, True, iter_count

        jac = np.array([[r_h, r_theta], [tangent[0], tangent[1]]], dtype=float)
        try:
            if np.linalg.cond(jac) > 1e14:
                return x, False, iter_count
            dx = np.linalg.solve(jac, -g)
        except np.linalg.LinAlgError:
            return x, False, iter_count

        alpha = 1.0
        norm_g_old = float(np.linalg.norm(g))
        accepted = False
        for _ls in range(25):
            x_trial = x + alpha * dx
            theta_trial = float(x_trial[1])
            if theta_trial < theta_min or theta_trial > theta_max:
                alpha *= 0.5
                continue
            r_trial, _, _ = model.residual_derivatives(float(x_trial[0]), theta_trial)
            g_trial = np.array([r_trial, np.dot(x_trial - x_pred, tangent)], dtype=float)
            norm_trial = float(np.linalg.norm(g_trial))
            if np.isfinite(norm_trial) and norm_trial < norm_g_old:
                x = x_trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            return x, False, iter_count

    h, theta = float(x[0]), float(x[1])
    r_final = model.residual(h, theta)
    arc_final = float(np.dot(x - x_pred, tangent))
    return x, (abs(r_final) < 1e-8 and abs(arc_final) < 1e-8), iter_count


def trace_pseudo_arclength_branch(
    model: ChiralEq10,
    state: ContinuationState,
    h0: float,
    theta0: float,
    h_lower: float,
    h_upper: float,
    theta_min: float,
    theta_max: float,
    ds: float,
    max_steps: int,
    newton_tol: float,
    max_newton_it: int,
    stable_tol: float,
):
    x = np.array([h0, theta0], dtype=float)
    tangent = compute_tangent(model, x[0], x[1])
    if tangent[0] < 0:
        tangent = -tangent

    h_values = [float(x[0])]
    theta_values = [float(x[1])]
    u_values = [float(model.energy(x[0], x[1]))]
    r_values = [float(model.residual(x[0], x[1]))]
    d2u_values = [float(model.second_derivative_energy(x[0], x[1]))]
    stable_values = [bool(d2u_values[-1] > stable_tol)]
    iter_values = [0]

    current_ds = ds
    for _step in range(max_steps):
        x_pred = x + current_ds * tangent
        x_new, success, iter_count = pseudo_arc_corrector(
            model, x_pred, tangent, theta_min, theta_max, newton_tol, max_newton_it
        )
        state.total_newton_iter += iter_count

        if not success:
            success_reduced = False
            ds_try = current_ds
            for _retry in range(8):
                ds_try *= 0.5
                x_pred_try = x + ds_try * tangent
                x_try, success_try, iter_try = pseudo_arc_corrector(
                    model, x_pred_try, tangent, theta_min, theta_max, newton_tol, max_newton_it
                )
                state.total_newton_iter += iter_try
                if success_try:
                    x_new = x_try
                    success_reduced = True
                    current_ds = ds_try
                    iter_count = iter_try
                    break
            if not success_reduced:
                log("Pseudo-arclength corrector failed. Stop this branch.")
                break

        h_new, theta_new = float(x_new[0]), float(x_new[1])
        if theta_new < theta_min - 1e-8 or theta_new > theta_max + 1e-8:
            log("Theta out of range. Stop this branch.")
            break
        if h_new < h_lower - 0.2 or h_new > h_upper + 0.2:
            log("h out of extended range. Stop this branch.")
            break

        h_values.append(h_new)
        theta_values.append(theta_new)
        u_values.append(float(model.energy(h_new, theta_new)))
        r_values.append(float(model.residual(h_new, theta_new)))
        d2u_values.append(float(model.second_derivative_energy(h_new, theta_new)))
        stable_values.append(bool(d2u_values[-1] > stable_tol))
        iter_values.append(iter_count)

        tangent_new = compute_tangent(model, h_new, theta_new)
        if float(np.dot(tangent_new, tangent)) < 0:
            tangent_new = -tangent_new
        x = x_new
        tangent = tangent_new
        current_ds = min(ds, current_ds * 1.2)

        if h_new >= h_upper:
            break

    return Branch(
        h=np.asarray(h_values, dtype=float),
        theta=np.asarray(theta_values, dtype=float),
        u=np.asarray(u_values, dtype=float),
        r=np.asarray(r_values, dtype=float),
        d2u=np.asarray(d2u_values, dtype=float),
        is_stable=np.asarray(stable_values, dtype=bool),
        newton_iters=np.asarray(iter_values, dtype=int),
    )


def interpolate_branch_to_samples(branch: Branch, h_sample, h_lower, h_upper):
    valid = (
        np.isfinite(branch.h)
        & np.isfinite(branch.theta)
        & (branch.h >= h_lower - 1e-10)
        & (branch.h <= h_upper + 1e-10)
    )
    h_b = branch.h[valid]
    theta_b = branch.theta[valid]
    u_b = branch.u[valid]
    r_b = branch.r[valid]
    d2u_b = branch.d2u[valid]
    if h_b.size < 2:
        nan = np.full_like(h_sample, np.nan, dtype=float)
        return nan.copy(), nan.copy(), nan.copy(), nan.copy()

    order = np.argsort(h_b)
    h_sort = h_b[order]
    theta_sort = theta_b[order]
    u_sort = u_b[order]
    r_sort = r_b[order]
    d2u_sort = d2u_b[order]

    h_unique, unique_idx = np.unique(h_sort, return_index=True)
    if h_unique.size < 2:
        nan = np.full_like(h_sample, np.nan, dtype=float)
        return nan.copy(), nan.copy(), nan.copy(), nan.copy()

    sample_idx = (h_sample >= h_unique[0]) & (h_sample <= h_unique[-1])
    theta_out = np.full_like(h_sample, np.nan, dtype=float)
    u_out = np.full_like(h_sample, np.nan, dtype=float)
    r_out = np.full_like(h_sample, np.nan, dtype=float)
    d2u_out = np.full_like(h_sample, np.nan, dtype=float)

    theta_out[sample_idx] = np.interp(h_sample[sample_idx], h_unique, theta_sort[unique_idx])
    u_out[sample_idx] = np.interp(h_sample[sample_idx], h_unique, u_sort[unique_idx])
    r_out[sample_idx] = np.interp(h_sample[sample_idx], h_unique, r_sort[unique_idx])
    d2u_out[sample_idx] = np.interp(h_sample[sample_idx], h_unique, d2u_sort[unique_idx])
    return theta_out, u_out, r_out, d2u_out


def maybe_plot(output_dir: Path, h_sample, branches, theta_selected, u_selected, r_selected, valid):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting
        log(f"Plotting skipped because matplotlib is unavailable: {exc}")
        return

    plt.rcParams["font.family"] = "Times New Roman"

    figure_path = output_dir / "Eq10_pseudo_arclength_energy_curve_python.png"
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(h_sample[valid], u_selected[valid], "r-", linewidth=2)
    ax.set_xlabel(r"$\bar{h}$")
    ax.set_ylabel(r"$\bar{U}$")
    ax.set_title("Pseudo-arclength continuation")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close()
    return figure_path


def main():
    parser = argparse.ArgumentParser(description="Pseudo-arclength continuation method using Eq. (10).")
    parser.add_argument("--a0-bar", type=float, default=0.8)
    parser.add_argument("--phi0-deg", type=float, default=50.0)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--h-lower", type=float, default=1e-3)
    parser.add_argument("--h-upper", type=float, default=3.0)
    parser.add_argument("--h-sample-number", "--h-number", dest="h_sample_number", type=int, default=2000)
    parser.add_argument("--ds", type=float, default=0.0005)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--newton-tol", type=float, default=1e-12)
    parser.add_argument("--max-newton-it", type=int, default=30)
    parser.add_argument("--theta-seed-number", type=int, default=2000)
    parser.add_argument("--root-unique-tol", type=float, default=1e-8)
    parser.add_argument("--stable-tol", type=float, default=1e-7)
    parser.add_argument("--track-only-min-energy-initial-root", action="store_true")
    parser.add_argument("--energy-factor", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    global VERBOSE
    VERBOSE = args.verbose

    args.output_dir.mkdir(parents=True, exist_ok=True)
    phi0 = math.radians(args.phi0_deg)
    model = ChiralEq10(args.a0_bar, phi0, args.n, args.energy_factor)
    state = ContinuationState()

    theta_min, theta_max = -math.pi, math.pi
    h_sample = np.linspace(args.h_lower, args.h_upper, args.h_sample_number)

    log("Input parameters")
    log("----------------")
    log(f"a0_bar    = {args.a0_bar:.10f}")
    log(f"phi0_deg  = {args.phi0_deg:.10f}")
    log(f"phi0_rad  = {phi0:.10f}")
    log(f"n         = {args.n}")
    log(f"h_lower   = {args.h_lower:.10e}")
    log(f"h_upper   = {args.h_upper:.10e}")
    log(f"theta_min = {theta_min:.10e}")
    log(f"theta_max = {theta_max:.10e}")
    log("\nComputed geometric parameters")
    log("-----------------------------")
    log(f"c_bar     = {model.c_bar:.10f}")
    log(f"b0_bar    = {model.b0_bar:.10f}")
    log(f"xi        = {model.xi:.10f}")
    log("\nPseudo-arclength settings")
    log("-------------------------")
    log(f"ds                         = {args.ds:.10e}")
    log(f"max_steps                  = {args.max_steps}")
    log(f"theta_seed_number          = {args.theta_seed_number}")
    log(f"track_all_stable_roots     = {not args.track_only_min_energy_initial_root}")
    log("\nPython / platform")
    log("-----------------")
    log(f"Python   = {platform.python_version()}")
    log(f"Platform = {platform.platform()}")

    h0 = args.h_lower
    theta_seeds = np.linspace(theta_min, theta_max, args.theta_seed_number)
    initial_roots = []
    for theta0 in theta_seeds:
        theta_root, success, iter_count = damped_newton_fixed_h(
            model, theta0, h0, theta_min, theta_max, args.newton_tol, args.max_newton_it
        )
        state.total_newton_iter += iter_count
        if success:
            initial_roots.append(theta_root)

    initial_roots = unique_roots(initial_roots, args.root_unique_tol)
    if initial_roots.size == 0:
        raise RuntimeError("No initial root found at h_lower. Increase theta_seed_number or change h_lower.")

    u_initial = np.array([model.energy(h0, theta) for theta in initial_roots], dtype=float)
    r_initial = np.array([model.residual(h0, theta) for theta in initial_roots], dtype=float)
    d2u_initial = np.array([model.second_derivative_energy(h0, theta) for theta in initial_roots], dtype=float)
    stable_initial = d2u_initial > args.stable_tol

    log("\nInitial roots at h_lower")
    log("------------------------")
    log(f"Number of initial roots        = {initial_roots.size}")
    log(f"Number of stable initial roots = {int(np.sum(stable_initial))}")
    for i, (theta, u, r, d2u, stable) in enumerate(
        zip(initial_roots, u_initial, r_initial, d2u_initial, stable_initial), start=1
    ):
        log(f"root {i}: theta = {theta:.12e}, U = {u:.12e}, R = {r:.3e}, d2U = {d2u:.3e}, stable = {int(stable)}")

    stable_idx = np.flatnonzero(stable_initial)
    if stable_idx.size == 0:
        log("Warning: no stable initial root found. All roots will be considered.")
        stable_idx = np.arange(initial_roots.size)

    if args.track_only_min_energy_initial_root:
        start_indices = np.array([stable_idx[int(np.argmin(u_initial[stable_idx]))]], dtype=int)
    else:
        start_indices = stable_idx

    log(f"\nNumber of branches to continue = {start_indices.size}")

    t0 = time.perf_counter()
    branches = []
    for branch_number, root_id in enumerate(start_indices, start=1):
        theta_start = float(initial_roots[root_id])
        log("\n============================================")
        log(f"Tracing branch {branch_number} / {start_indices.size}")
        log(f"Initial theta = {theta_start:.12e}")
        log("============================================")
        branch = trace_pseudo_arclength_branch(
            model,
            state,
            h0,
            theta_start,
            args.h_lower,
            args.h_upper,
            theta_min,
            theta_max,
            args.ds,
            args.max_steps,
            args.newton_tol,
            args.max_newton_it,
            args.stable_tol,
        )
        branches.append(branch)
        log(f"Branch {branch_number}: points = {branch.h.size}, reached h_max = {np.nanmax(branch.h):.8f}")
    elapsed = time.perf_counter() - t0

    num_branches = len(branches)
    theta_branch_sample = np.full((args.h_sample_number, num_branches), np.nan)
    u_branch_sample = np.full((args.h_sample_number, num_branches), np.nan)
    r_branch_sample = np.full((args.h_sample_number, num_branches), np.nan)
    d2u_branch_sample = np.full((args.h_sample_number, num_branches), np.nan)

    for branch_id, branch in enumerate(branches):
        theta_out, u_out, r_out, d2u_out = interpolate_branch_to_samples(
            branch, h_sample, args.h_lower, args.h_upper
        )
        theta_branch_sample[:, branch_id] = theta_out
        u_branch_sample[:, branch_id] = u_out
        r_branch_sample[:, branch_id] = r_out
        d2u_branch_sample[:, branch_id] = d2u_out

    theta_selected = np.full(args.h_sample_number, np.nan)
    u_selected = np.full(args.h_sample_number, np.nan)
    r_selected = np.full(args.h_sample_number, np.nan)
    d2u_selected = np.full(args.h_sample_number, np.nan)
    selected_branch_id = np.full(args.h_sample_number, np.nan)

    for i in range(args.h_sample_number):
        u_i = u_branch_sample[i, :]
        theta_i = theta_branch_sample[i, :]
        r_i = r_branch_sample[i, :]
        d2u_i = d2u_branch_sample[i, :]
        valid_i = np.isfinite(u_i) & np.isfinite(theta_i) & np.isfinite(r_i)
        if not np.any(valid_i):
            continue
        stable_i = valid_i & (d2u_i > args.stable_tol)
        if np.any(stable_i):
            candidate_idx = np.flatnonzero(stable_i)
        else:
            candidate_idx = np.flatnonzero(valid_i)
        selected_id = candidate_idx[int(np.argmin(u_i[candidate_idx]))]
        theta_selected[i] = theta_i[selected_id]
        u_selected[i] = u_i[selected_id]
        r_selected[i] = r_i[selected_id]
        d2u_selected[i] = d2u_i[selected_id]
        selected_branch_id[i] = selected_id + 1  # MATLAB-style 1-based branch id.

    valid_selected = np.isfinite(theta_selected)
    for i in np.flatnonzero(valid_selected):
        r_selected[i] = model.residual(h_sample[i], theta_selected[i])
        u_selected[i] = model.energy(h_sample[i], theta_selected[i])
        d2u_selected[i] = model.second_derivative_energy(h_sample[i], theta_selected[i])

    valid_selected = np.isfinite(theta_selected) & np.isfinite(r_selected)
    l_g_arc = float(np.mean(r_selected[valid_selected] ** 2)) if np.any(valid_selected) else math.nan
    rmax_arc = float(np.max(np.abs(r_selected[valid_selected]))) if np.any(valid_selected) else math.nan

    log("\n\n============================================")
    log("Pseudo-arclength continuation result")
    log("============================================")
    log(f"Valid sampled points     = {np.sum(valid_selected)} / {args.h_sample_number}")
    log(f"L_G = mean(R_E^2)        = {l_g_arc:.12e}")
    log(f"R_E max                  = {rmax_arc:.12e}")
    log(f"Total Newton iterations  = {state.total_newton_iter}")
    log(f"Residual evaluations     = {model.feval_count}")
    log(f"Elapsed time             = {elapsed:.6f} s")

    if args.save_csv:
        selected_file = args.output_dir / "Eq10_pseudo_arclength_selected_path_python.csv"
        selected_data = np.column_stack(
            [
                h_sample,
                theta_selected,
                u_selected,
                r_selected,
                r_selected**2,
                d2u_selected,
                selected_branch_id,
            ]
        )
        np.savetxt(
            selected_file,
            selected_data,
            delimiter=",",
            header="h_bar,theta_arc,U_arc,R_E_arc,R_E_squared_arc,d2U_dtheta2,selected_branch_id",
            comments="",
        )

        summary_file = args.output_dir / "Eq10_pseudo_arclength_summary_python.csv"
        with summary_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Method",
                    "L_G",
                    "R_E_max",
                    "Total_Newton_iterations",
                    "Residual_evaluations",
                    "Elapsed_time_s",
                    "Valid_points",
                    "Num_branches",
                ]
            )
            writer.writerow(
                [
                    "Pseudo_arclength",
                    l_g_arc,
                    rmax_arc,
                    state.total_newton_iter,
                    model.feval_count,
                    elapsed,
                    int(np.sum(valid_selected)),
                    num_branches,
                ]
            )

        all_branch_file = args.output_dir / "Eq10_pseudo_arclength_all_branches_python.csv"
        with all_branch_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["branch_id", "point_id", "h_bar", "theta", "U", "R_E", "d2U_dtheta2", "is_stable", "newton_iterations"]
            )
            for branch_id, branch in enumerate(branches, start=1):
                for point_id in range(branch.h.size):
                    writer.writerow(
                        [
                            branch_id,
                            point_id + 1,
                            branch.h[point_id],
                            branch.theta[point_id],
                            branch.u[point_id],
                            branch.r[point_id],
                            branch.d2u[point_id],
                            int(branch.is_stable[point_id]),
                            int(branch.newton_iters[point_id]),
                        ]
                    )

    figure_path = None
    if not args.no_plot:
        figure_path = maybe_plot(args.output_dir, h_sample, branches, theta_selected, u_selected, r_selected, valid_selected)

    print(f"Elapsed_time_s = {elapsed:.6f}")
    if figure_path is not None:
        print(f"Energy_figure = {figure_path}")


if __name__ == "__main__":
    main()
