from __future__ import annotations
import argparse
import csv
import math
import platform
import time
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

    def energy(self, theta, h_bar):
        a_d_bar, b_d_bar = self.deformed_lengths(h_bar, theta, self.n)
        return self.energy_factor * self.n * (
            (a_d_bar - self.a0_bar) ** 2 + self.xi * (b_d_bar - self.b0_bar) ** 2
        )

    def residual(self, theta, h_bar):
        theta_arr = np.asarray(theta, dtype=float)
        self.feval_count += theta_arr.size
        a_d_bar, b_d_bar = self.deformed_lengths(h_bar, theta_arr, self.n)
        eps_val = 1e-14
        a_d_bar = np.maximum(a_d_bar, eps_val)
        b_d_bar = np.maximum(b_d_bar, eps_val)
        r = (
            (1.0 - self.a0_bar / a_d_bar) * np.sin(theta_arr)
            + self.xi
            * (1.0 - self.b0_bar / b_d_bar)
            * np.sin(theta_arr + 2.0 * math.pi / self.n)
        )
        if np.ndim(theta) == 0:
            return float(r)
        return r

    def residual_and_derivative(self, theta, h_bar):
        r = self.residual(theta, h_bar)
        delta = 2.0 * math.pi / self.n
        a_d_bar, b_d_bar = self.deformed_lengths(h_bar, theta, self.n)
        eps_val = 1e-14
        a_d_bar = np.maximum(a_d_bar, eps_val)
        b_d_bar = np.maximum(b_d_bar, eps_val)
        d_r = (
            self.a0_bar * np.sin(theta) ** 2 / a_d_bar**3
            + (1.0 - self.a0_bar / a_d_bar) * np.cos(theta)
            + self.xi
            * (
                self.b0_bar * np.sin(theta + delta) ** 2 / b_d_bar**3
                + (1.0 - self.b0_bar / b_d_bar) * np.cos(theta + delta)
            )
        )
        if np.ndim(theta) == 0:
            return float(r), float(d_r)
        return r, d_r

    def second_derivative_energy(self, theta, h_bar, dtheta: float = 1e-5):
        theta_p = min(theta + dtheta, math.pi)
        theta_m = max(theta - dtheta, -math.pi)
        if abs(theta_p - theta_m) < 1e-14:
            return math.nan
        u0 = self.energy(theta, h_bar)
        up = self.energy(theta_p, h_bar)
        um = self.energy(theta_m, h_bar)
        return float((up - 2.0 * u0 + um) / (dtheta**2))


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


def damped_newton_eq10(
    model: ChiralEq10,
    theta0: float,
    h_bar: float,
    theta_min: float,
    theta_max: float,
    tol: float,
    max_iter: int,
):
    theta = min(max(float(theta0), theta_min), theta_max)
    for _ in range(max_iter):
        r, d_r = model.residual_and_derivative(theta, h_bar)
        if abs(r) < tol:
            return theta, True
        if (not np.isfinite(r)) or (not np.isfinite(d_r)) or abs(d_r) < 1e-14:
            return theta, False

        step = -r / d_r
        alpha = 1.0
        r_old = abs(r)
        accepted = False
        for _ls in range(25):
            theta_trial = theta + alpha * step
            if theta_trial < theta_min or theta_trial > theta_max:
                alpha *= 0.5
                continue
            r_trial = abs(model.residual(theta_trial, h_bar))
            if np.isfinite(r_trial) and r_trial < r_old:
                theta = theta_trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            return theta, False

        if abs(alpha * step) < tol:
            r_final = abs(model.residual(theta, h_bar))
            return theta, r_final < 1e-8

    return theta, abs(model.residual(theta, h_bar)) < 1e-8


def damped_newton_eq10_many(
    model: ChiralEq10,
    theta0_array,
    h_bar: float,
    theta_min: float,
    theta_max: float,
    tol: float,
    max_iter: int,
):
    """Vectorized damped Newton iteration for all seed values at one height."""
    theta = np.clip(np.asarray(theta0_array, dtype=float).copy(), theta_min, theta_max)
    active = np.ones(theta.shape, dtype=bool)
    success = np.zeros(theta.shape, dtype=bool)

    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break

        r, d_r = model.residual_and_derivative(theta[idx], h_bar)
        converged = np.abs(r) < tol
        if np.any(converged):
            success[idx[converged]] = True
            active[idx[converged]] = False

        keep = ~converged
        if not np.any(keep):
            continue

        idx_keep = idx[keep]
        r_keep = r[keep]
        d_keep = d_r[keep]
        invalid = (~np.isfinite(r_keep)) | (~np.isfinite(d_keep)) | (np.abs(d_keep) < 1e-14)
        if np.any(invalid):
            active[idx_keep[invalid]] = False

        valid_newton = ~invalid
        if not np.any(valid_newton):
            continue

        idx_newton = idx_keep[valid_newton]
        r_newton = r_keep[valid_newton]
        d_newton = d_keep[valid_newton]
        step = -r_newton / d_newton
        alpha = np.ones_like(step)
        r_old = np.abs(r_newton)
        accepted = np.zeros_like(step, dtype=bool)

        for _ls in range(25):
            pending = ~accepted
            if not np.any(pending):
                break

            theta_trial = theta[idx_newton[pending]] + alpha[pending] * step[pending]
            in_bounds = (theta_trial >= theta_min) & (theta_trial <= theta_max)

            pending_indices = np.flatnonzero(pending)
            out_indices = pending_indices[~in_bounds]
            if out_indices.size:
                alpha[out_indices] *= 0.5

            in_indices = pending_indices[in_bounds]
            if in_indices.size:
                theta_trial_in = theta[idx_newton[in_indices]] + alpha[in_indices] * step[in_indices]
                r_trial = np.abs(model.residual(theta_trial_in, h_bar))
                ok = np.isfinite(r_trial) & (r_trial < r_old[in_indices])
                if np.any(ok):
                    accept_indices = in_indices[ok]
                    theta[idx_newton[accept_indices]] = theta_trial_in[ok]
                    accepted[accept_indices] = True
                reject_indices = in_indices[~ok]
                if reject_indices.size:
                    alpha[reject_indices] *= 0.5

        if np.any(~accepted):
            active[idx_newton[~accepted]] = False

        if np.any(accepted):
            accepted_indices = np.flatnonzero(accepted)
            small_step = np.abs(alpha[accepted_indices] * step[accepted_indices]) < tol
            if np.any(small_step):
                small_indices = accepted_indices[small_step]
                final_roots = idx_newton[small_indices]
                r_final = np.abs(model.residual(theta[final_roots], h_bar))
                ok_final = r_final < 1e-8
                success[final_roots[ok_final]] = True
                active[final_roots] = False

    idx = np.flatnonzero(active)
    if idx.size:
        r_final = np.abs(model.residual(theta[idx], h_bar))
        ok_final = r_final < 1e-8
        success[idx[ok_final]] = True

    return theta[success]


def maybe_plot(output_dir: Path, h_array, theta_selected, u_selected, r_selected, valid):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting
        log(f"Plotting skipped because matplotlib is unavailable: {exc}")
        return

    plt.rcParams["font.family"] = "Times New Roman"

    figure_path = output_dir / "Eq10_Newton_energy_curve_python.png"
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(h_array[valid], u_selected[valid], "r-", linewidth=2)
    ax.set_xlabel(r"$\bar{h}$")
    ax.set_ylabel(r"$\bar{U}$")
    ax.set_title("Newton method")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close()
    return figure_path


def main():
    parser = argparse.ArgumentParser(description="Newton root-finding method using Eq. (10).")
    parser.add_argument("--a0-bar", type=float, default=0.8)
    parser.add_argument("--phi0-deg", type=float, default=50.0)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--h-lower", type=float, default=1e-3)
    parser.add_argument("--h-upper", type=float, default=3.0)
    parser.add_argument("--h-number", type=int, default=2000)
    parser.add_argument("--theta-seed-number", type=int, default=2000)
    parser.add_argument("--newton-tol", type=float, default=1e-12)
    parser.add_argument("--max-newton-it", type=int, default=50)
    parser.add_argument("--root-unique-tol", type=float, default=1e-7)
    parser.add_argument("--stable-tol", type=float, default=1e-7)
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

    h_array = np.linspace(args.h_lower, args.h_upper, args.h_number)
    theta_min, theta_max = -math.pi, math.pi
    theta_seeds = np.linspace(theta_min, theta_max, args.theta_seed_number)

    log("Input parameters")
    log("----------------")
    log(f"a0_bar   = {args.a0_bar:.10f}")
    log(f"phi0_deg = {args.phi0_deg:.10f}")
    log(f"phi0_rad = {phi0:.10f}")
    log(f"n        = {args.n}")
    log(f"c_bar    = {model.c_bar:.10f}")
    log(f"b0_bar   = {model.b0_bar:.10f}")
    log(f"xi       = {model.xi:.10f}")
    log("\nNewton settings")
    log("---------------")
    log(f"h_number          = {args.h_number}")
    log(f"theta_seed_number = {args.theta_seed_number}")
    log(f"newton_tol        = {args.newton_tol:.3e}")
    log("\nPython / platform")
    log("-----------------")
    log(f"Python   = {platform.python_version()}")
    log(f"Platform = {platform.platform()}")

    theta_selected = np.full(args.h_number, np.nan)
    u_selected = np.full(args.h_number, np.nan)
    r_selected = np.full(args.h_number, np.nan)
    d2u_selected = np.full(args.h_number, np.nan)
    num_roots_all = np.zeros(args.h_number, dtype=int)
    num_stable_all = np.zeros(args.h_number, dtype=int)
    selected_stable = np.zeros(args.h_number, dtype=int)
    all_root_rows = []

    t0 = time.perf_counter()
    for i, h_i in enumerate(h_array, start=1):
        candidate_roots = damped_newton_eq10_many(
            model,
            theta_seeds,
            h_i,
            theta_min,
            theta_max,
            args.newton_tol,
            args.max_newton_it,
        )
        candidate_roots = unique_roots(candidate_roots, args.root_unique_tol)
        candidate_roots = candidate_roots[
            (candidate_roots >= theta_min - 1e-8) & (candidate_roots <= theta_max + 1e-8)
        ]
        num_roots_all[i - 1] = candidate_roots.size
        if candidate_roots.size == 0:
            continue

        u_candidates = np.array([model.energy(theta, h_i) for theta in candidate_roots], dtype=float)
        r_candidates = np.array([model.residual(theta, h_i) for theta in candidate_roots], dtype=float)
        d2u_candidates = np.array(
            [model.second_derivative_energy(theta, h_i) for theta in candidate_roots], dtype=float
        )
        stable_flags = d2u_candidates > args.stable_tol
        num_stable_all[i - 1] = int(np.sum(stable_flags))

        if np.any(stable_flags):
            stable_idx = np.flatnonzero(stable_flags)
            selected_id = stable_idx[int(np.argmin(u_candidates[stable_idx]))]
            selected_stable[i - 1] = 1
        else:
            selected_id = int(np.argmin(u_candidates))
            selected_stable[i - 1] = 0

        theta_selected[i - 1] = candidate_roots[selected_id]
        u_selected[i - 1] = u_candidates[selected_id]
        r_selected[i - 1] = r_candidates[selected_id]
        d2u_selected[i - 1] = d2u_candidates[selected_id]

        for theta, u, r, d2u, stable in zip(
            candidate_roots, u_candidates, r_candidates, d2u_candidates, stable_flags
        ):
            all_root_rows.append((i, h_i, theta, u, r, d2u, int(stable)))

    elapsed = time.perf_counter() - t0
    valid = np.isfinite(theta_selected) & np.isfinite(r_selected)
    l_e = float(np.mean(r_selected[valid] ** 2)) if np.any(valid) else math.nan
    r_max = float(np.max(np.abs(r_selected[valid]))) if np.any(valid) else math.nan

    log("\n\n============================================")
    log("Newton root-finding result")
    log("============================================")
    log(f"Valid height points     = {np.sum(valid)} / {args.h_number}")
    log(f"L_E = mean(R_E^2)       = {l_e:.12e}")
    log(f"R_E max                 = {r_max:.12e}")
    log(f"Mean number of roots    = {np.mean(num_roots_all[valid]):.6f}" if np.any(valid) else "Mean number of roots    = nan")
    log(f"Mean stable roots       = {np.mean(num_stable_all[valid]):.6f}" if np.any(valid) else "Mean stable roots       = nan")
    log(f"Residual evaluations    = {model.feval_count}")
    log(f"Elapsed time            = {elapsed:.6f} s")

    if args.save_csv:
        selected_file = args.output_dir / "Eq10_Newton_root_finding_result_python.csv"
        selected_data = np.column_stack(
            [
                h_array,
                theta_selected,
                u_selected,
                r_selected,
                r_selected**2,
                d2u_selected,
                num_roots_all,
                num_stable_all,
                selected_stable,
            ]
        )
        np.savetxt(
            selected_file,
            selected_data,
            delimiter=",",
            header="h_bar,theta_Newton,U_Newton,R_E_Newton,R_E_squared_Newton,d2U_dtheta2,num_roots,num_stable_roots,selected_is_stable",
            comments="",
        )

        all_roots_file = args.output_dir / "Eq10_Newton_all_root_candidates_python.csv"
        with all_roots_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["h_index", "h_bar", "theta", "U", "R_E", "d2U_dtheta2", "is_stable"])
            writer.writerows(all_root_rows)

        summary_file = args.output_dir / "Eq10_Newton_summary_python.csv"
        with summary_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Method",
                    "L_E",
                    "R_E_max",
                    "Residual_evaluations",
                    "Elapsed_time_s",
                    "Valid_points",
                    "Mean_num_roots",
                    "Mean_num_stable_roots",
                ]
            )
            writer.writerow(
                [
                    "Newton",
                    l_e,
                    r_max,
                    model.feval_count,
                    elapsed,
                    int(np.sum(valid)),
                    float(np.mean(num_roots_all[valid])) if np.any(valid) else math.nan,
                    float(np.mean(num_stable_all[valid])) if np.any(valid) else math.nan,
                ]
            )

    figure_path = None
    if not args.no_plot:
        figure_path = maybe_plot(args.output_dir, h_array, theta_selected, u_selected, r_selected, valid)

    print(f"Elapsed_time_s = {elapsed:.6f}")
    if figure_path is not None:
        print(f"Energy_figure = {figure_path}")


if __name__ == "__main__":
    main()
