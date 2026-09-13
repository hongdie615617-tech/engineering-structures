from __future__ import annotations
import argparse
import csv
import math
import platform
import time
from pathlib import Path

import numpy as np



def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


class ChiralEq10:
    def __init__(self, a0_bar: float, phi0: float, n: int):
        self.a0_bar = float(a0_bar)
        self.phi0 = float(phi0)
        self.n = int(n)
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

    def compute_u_and_residual(self, theta_vec, h_array):
        theta_vec = np.asarray(theta_vec, dtype=float).reshape(-1)
        h_array = np.asarray(h_array, dtype=float).reshape(-1)
        a_d_bar, b_d_bar = self.deformed_lengths(h_array, theta_vec, self.n)
        u_bar = self.n * ((a_d_bar - self.a0_bar) ** 2 + self.xi * (b_d_bar - self.b0_bar) ** 2)

        eps_val = 1e-14
        a_d_bar = np.maximum(a_d_bar, eps_val)
        b_d_bar = np.maximum(b_d_bar, eps_val)
        residual = (
            (1.0 - self.a0_bar / a_d_bar) * np.sin(theta_vec)
            + self.xi
            * (1.0 - self.b0_bar / b_d_bar)
            * np.sin(theta_vec + 2.0 * math.pi / self.n)
        )
        return u_bar, residual

    def residual_derivative_theta(self, theta_vec, h_array):
        theta_vec = np.asarray(theta_vec, dtype=float).reshape(-1)
        h_array = np.asarray(h_array, dtype=float).reshape(-1)
        delta = 2.0 * math.pi / self.n
        a_d_bar, b_d_bar = self.deformed_lengths(h_array, theta_vec, self.n)
        eps_val = 1e-14
        a_d_bar = np.maximum(a_d_bar, eps_val)
        b_d_bar = np.maximum(b_d_bar, eps_val)
        return (
            self.a0_bar * np.sin(theta_vec) ** 2 / a_d_bar**3
            + (1.0 - self.a0_bar / a_d_bar) * np.cos(theta_vec)
            + self.xi
            * (
                self.b0_bar * np.sin(theta_vec + delta) ** 2 / b_d_bar**3
                + (1.0 - self.b0_bar / b_d_bar) * np.cos(theta_vec + delta)
            )
        )


def dgo_forward_objective(theta_vec, h_array, model: ChiralEq10, lambda_e, lambda_p, lambda_s):
    u_bar, residual = model.compute_u_and_residual(theta_vec, h_array)
    l_e = float(np.mean(residual**2))
    l_p = float(np.mean(u_bar))
    if lambda_s > 0 and len(theta_vec) >= 3:
        d2theta = np.diff(theta_vec, n=2)
        l_s = float(np.mean(d2theta**2))
    else:
        l_s = 0.0
    return lambda_e * l_e + lambda_p * l_p + lambda_s * l_s


def dgo_forward_gradient(theta_vec, h_array, model: ChiralEq10, lambda_e, lambda_p, lambda_s):
    theta_vec = np.asarray(theta_vec, dtype=float).reshape(-1)
    _u_bar, residual = model.compute_u_and_residual(theta_vec, h_array)
    d_residual = model.residual_derivative_theta(theta_vec, h_array)
    m = theta_vec.size

    # dU/dtheta = 2*n*R_E for this nondimensional energy model.
    grad = lambda_e * (2.0 * residual * d_residual / m) + lambda_p * (2.0 * model.n * residual / m)

    if lambda_s > 0 and m >= 3:
        d2theta = np.diff(theta_vec, n=2)
        denom = d2theta.size
        grad_s = np.zeros_like(theta_vec)
        grad_s[:-2] += 2.0 * d2theta / denom
        grad_s[1:-1] += -4.0 * d2theta / denom
        grad_s[2:] += 2.0 * d2theta / denom
        grad += lambda_s * grad_s
    return grad



def order_penalty_and_gradient(theta_vec, order_mode: str, lambda_order: float):
    """Soft monotonicity penalty used by the fast L-BFGS-B mode."""
    theta_vec = np.asarray(theta_vec, dtype=float).reshape(-1)
    if order_mode == "none" or lambda_order <= 0 or theta_vec.size < 2:
        return 0.0, np.zeros_like(theta_vec)

    diff = np.diff(theta_vec)
    if order_mode == "descend":
        violation = np.maximum(diff, 0.0)
    elif order_mode == "ascend":
        violation = np.maximum(-diff, 0.0)
    else:
        raise ValueError("Unknown order_mode. Use descend, ascend, or none.")

    denom = max(1, violation.size)
    penalty = lambda_order * float(np.mean(violation**2))
    grad = np.zeros_like(theta_vec)
    active = violation > 0
    coeff = 2.0 * lambda_order * violation / denom
    if order_mode == "descend":
        grad[:-1] -= coeff * active
        grad[1:] += coeff * active
    else:
        grad[:-1] += coeff * active
        grad[1:] -= coeff * active
    return penalty, grad


def build_order_constraint(h_number: int, order_mode: str):
    if order_mode == "none":
        return []
    if h_number < 2:
        return []

    if order_mode == "descend":
        # theta_{i+1} - theta_i <= 0
        mat = diags([-np.ones(h_number - 1), np.ones(h_number - 1)], [0, 1], shape=(h_number - 1, h_number))
    elif order_mode == "ascend":
        # theta_i - theta_{i+1} <= 0
        mat = diags([np.ones(h_number - 1), -np.ones(h_number - 1)], [0, 1], shape=(h_number - 1, h_number))
    else:
        raise ValueError("Unknown order_mode. Use descend, ascend, or none.")
    return [LinearConstraint(mat, -np.inf, np.zeros(h_number - 1))]


def initial_guess(h_number: int, init_mode: str, order_mode: str, rng: np.random.Generator):
    if init_mode == "linear_descend":
        theta0 = np.linspace(0.9 * math.pi, -0.9 * math.pi, h_number)
    elif init_mode == "linear_weak_descend":
        theta0 = np.linspace(0.5 * math.pi, -0.5 * math.pi, h_number)
    elif init_mode == "linear_zero_to_negative":
        theta0 = np.linspace(0.0, -0.9 * math.pi, h_number)
    elif init_mode == "linear_positive_to_zero":
        theta0 = np.linspace(0.9 * math.pi, 0.0, h_number)
    elif init_mode == "constant_negative":
        theta0 = -0.2 * np.ones(h_number)
    elif init_mode == "zero":
        theta0 = np.zeros(h_number)
    elif init_mode == "random":
        theta0 = -math.pi + 2.0 * math.pi * rng.random(h_number)
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    if order_mode == "descend":
        theta0 = np.sort(theta0)[::-1]
    elif order_mode == "ascend":
        theta0 = np.sort(theta0)
    return theta0


def initial_guess_list(h_number: int, order_mode: str, seed: int, random_starts: int):
    rng = np.random.default_rng(seed)
    modes = [
        "linear_descend",
        "linear_weak_descend",
        "linear_zero_to_negative",
        "linear_positive_to_zero",
        "constant_negative",
        "zero",
    ]
    starts = [(mode, initial_guess(h_number, mode, order_mode, rng)) for mode in modes]
    for idx in range(max(0, random_starts)):
        starts.append((f"random_{idx + 1:02d}", initial_guess(h_number, "random", order_mode, rng)))
    return starts


def case_parameters(case_name: str):
    if case_name == "case1":
        return 0.8, 50.0, 6
    if case_name == "case2":
        return 2.0, math.degrees(0.4887), 6
    if case_name == "case3":
        return 1.5, 30.0, 6
    raise ValueError("Unknown case. Use case1, case2, or case3.")


def maybe_plot(output_dir: Path, h_array, theta, u_bar, residual, label: str):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        log(f"Plotting skipped because matplotlib is unavailable: {exc}")
        return

    plt.rcParams["font.family"] = "Times New Roman"

    figure_path = output_dir / "DGO_forward_energy_curve_python.png"
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(h_array, u_bar, "r-", linewidth=2)
    ax.set_xlabel(r"$\bar{h}$")
    ax.set_ylabel(r"$\bar{U}$")
    ax.set_title("Direct gradient optimization")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close()
    return figure_path


def main():
    parser = argparse.ArgumentParser(description="Direct gradient optimization using Eq. (10).")
    parser.add_argument("--case", choices=["case1", "case2", "case3"], default="case1")
    parser.add_argument("--a0", "--a0-bar", dest="a0_bar", type=float, default=None)
    parser.add_argument("--phi0", "--phi0-deg", dest="phi0_deg", type=float, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--h-lower", type=float, default=1e-3)
    parser.add_argument("--h-upper", type=float, default=3.0)
    parser.add_argument("--h-number", type=int, default=2000)
    parser.add_argument("--lambda-e", type=float, default=1.0)
    parser.add_argument("--lambda-p", type=float, default=1.0)
    parser.add_argument("--lambda-s", type=float, default=0.0)
    parser.add_argument("--order-mode", choices=["descend", "ascend", "none"], default="descend")
    parser.add_argument("--optimizer", choices=["lbfgsb", "slsqp"], default="lbfgsb")
    parser.add_argument("--lambda-order", type=float, default=1000.0)
    parser.add_argument("--single-start", action="store_true")
    parser.add_argument("--init-mode", default="linear_descend")
    parser.add_argument("--starts", type=int, default=None, help="Maximum number of starts in multi-start mode.")
    parser.add_argument("--random-starts", type=int, default=24, help="Number of random initial paths added to the deterministic starts.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-fun-eval", type=int, default=500000)
    parser.add_argument("--ftol", type=float, default=1e-12)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    global VERBOSE
    VERBOSE = args.verbose

    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_a0, case_phi0_deg, case_n = case_parameters(args.case)
    a0_bar = args.a0_bar if args.a0_bar is not None else case_a0
    phi0_deg = args.phi0_deg if args.phi0_deg is not None else case_phi0_deg
    n = args.n if args.n is not None else case_n
    phi0 = math.radians(phi0_deg)

    model = ChiralEq10(a0_bar, phi0, n)
    h_array = np.linspace(args.h_lower, args.h_upper, args.h_number)

    log("Input parameters")
    log("----------------")
    log(f"case     = {args.case}")
    log(f"a0_bar   = {a0_bar:.10f}")
    log(f"phi0     = {phi0:.10f} rad")
    log(f"phi0_deg = {phi0_deg:.10f} deg")
    log(f"n        = {n}")
    log("\nComputed geometric parameters")
    log("-----------------------------")
    log(f"c_bar    = {model.c_bar:.10f}")
    log(f"b0_bar   = {model.b0_bar:.10f}")
    log(f"xi       = {model.xi:.10f}")
    log("\nOptimization settings")
    log("---------------------")
    log(f"h_number   = {args.h_number}")
    log(f"lambda_E   = {args.lambda_e}")
    log(f"lambda_P   = {args.lambda_p}")
    log(f"lambda_S   = {args.lambda_s}")
    log(f"order_mode = {args.order_mode}")
    log(f"optimizer  = scipy {args.optimizer}")
    log("\nPython / platform")
    log("-----------------")
    log(f"Python   = {platform.python_version()}")
    log(f"Platform = {platform.platform()}")

    bounds = Bounds(-math.pi * np.ones(args.h_number), math.pi * np.ones(args.h_number))
    if args.optimizer == "slsqp":
        constraints = build_order_constraint(args.h_number, args.order_mode)
        options = {
            "maxiter": args.max_iter,
            "ftol": args.ftol,
            "eps": args.eps,
            "disp": bool(args.verbose),
        }
    else:
        constraints = []
        options = {
            "maxiter": args.max_iter,
            "ftol": args.ftol,
            "disp": bool(args.verbose),
            "maxls": 50,
        }

    if args.single_start:
        rng = np.random.default_rng(args.seed)
        starts = [(args.init_mode, initial_guess(args.h_number, args.init_mode, args.order_mode, rng))]
    else:
        starts = initial_guess_list(args.h_number, args.order_mode, args.seed, args.random_starts)
        if args.starts is not None:
            starts = starts[: max(1, args.starts)]

    results = []
    t0 = time.perf_counter()
    for start_id, (mode, theta0) in enumerate(starts, start=1):
        log("\n============================================")
        log(f"Start {start_id} / {len(starts)}")
        log(f"Initial mode = {mode}")
        log("============================================")

        def fun(theta_vec):
            value = dgo_forward_objective(theta_vec, h_array, model, args.lambda_e, args.lambda_p, args.lambda_s)
            if args.optimizer == "lbfgsb":
                value += order_penalty_and_gradient(theta_vec, args.order_mode, args.lambda_order)[0]
            return value

        def jac(theta_vec):
            grad = dgo_forward_gradient(theta_vec, h_array, model, args.lambda_e, args.lambda_p, args.lambda_s)
            if args.optimizer == "lbfgsb":
                grad = grad + order_penalty_and_gradient(theta_vec, args.order_mode, args.lambda_order)[1]
            return grad

        result = minimize(
            fun,
            theta0,
            jac=jac,
            method="SLSQP" if args.optimizer == "slsqp" else "L-BFGS-B",
            bounds=bounds,
            constraints=constraints,
            options=options,
        )
        theta_opt = result.x
        u_opt, r_opt = model.compute_u_and_residual(theta_opt, h_array)
        l_e = float(np.mean(r_opt**2))
        l_g = l_e
        l_p = float(np.mean(u_opt))
        l_total = args.lambda_e * l_e + args.lambda_p * l_p
        if args.lambda_s > 0 and theta_opt.size >= 3:
            l_total += args.lambda_s * float(np.mean(np.diff(theta_opt, n=2) ** 2))
        if args.optimizer == "lbfgsb":
            l_total += order_penalty_and_gradient(theta_opt, args.order_mode, args.lambda_order)[0]
        r_max = float(np.max(np.abs(r_opt)))
        results.append(
            {
                "start_id": start_id,
                "mode": mode,
                "theta": theta_opt,
                "u": u_opt,
                "r": r_opt,
                "result": result,
                "L_E": l_e,
                "L_G": l_g,
                "L_P": l_p,
                "L_total": l_total,
                "R_max": r_max,
            }
        )
        log(f"success     = {result.success}")
        log(f"message     = {result.message}")
        log(f"fval        = {result.fun:.12e}")
        log(f"L_G = L_E   = {l_g:.12e}")
        log(f"L_P         = {l_p:.12e}")
        log(f"L_total     = {l_total:.12e}")
        log(f"R_max       = {r_max:.12e}")
        log(f"iterations  = {getattr(result, 'nit', math.nan)}")
        log(f"func evals  = {getattr(result, 'nfev', math.nan)}")

    elapsed = time.perf_counter() - t0
    best = min(results, key=lambda item: item["L_total"])

    log("\n\n============================================")
    log("Best standard gradient-based optimization result")
    log("============================================")
    log(f"Best start ID = {best['start_id']}")
    log(f"Best mode     = {best['mode']}")
    log(f"L_G = L_E     = {best['L_G']:.12e}")
    log(f"L_P           = {best['L_P']:.12e}")
    log(f"L_total       = {best['L_total']:.12e}")
    log(f"R_max         = {best['R_max']:.12e}")
    log(f"Elapsed time  = {elapsed:.6f} s")

    if args.save_csv:
        if args.single_start:
            output_file = args.output_dir / "DGO_forward_single_start_result_python.csv"
        else:
            output_file = args.output_dir / "DGO_forward_PINN_consistent_loss_python.csv"
        output_data = np.column_stack([h_array, best["theta"], best["u"], best["r"], best["r"] ** 2])
        np.savetxt(
            output_file,
            output_data,
            delimiter=",",
            header="h_bar,theta_gradient,U_gradient,R_E_gradient,R_E_squared_gradient",
            comments="",
        )

        summary_file = args.output_dir / "DGO_forward_summary_python.csv"
        with summary_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Method",
                    "case",
                    "single_start",
                    "best_start_id",
                    "best_mode",
                    "L_G",
                    "L_P",
                    "L_total",
                    "R_max",
                    "Elapsed_time_s",
                    "h_number",
                    "success",
                    "message",
                    "iterations",
                    "func_evals",
                ]
            )
            res = best["result"]
            writer.writerow(
                [
                    "DGO_SLSQP",
                    args.case,
                    int(args.single_start),
                    best["start_id"],
                    best["mode"],
                    best["L_G"],
                    best["L_P"],
                    best["L_total"],
                    best["R_max"],
                    elapsed,
                    args.h_number,
                    int(res.success),
                    res.message,
                    getattr(res, "nit", math.nan),
                    getattr(res, "nfev", math.nan),
                ]
            )

    figure_path = None
    if not args.no_plot:
        figure_path = maybe_plot(args.output_dir, h_array, best["theta"], best["u"], best["r"], "DGO_forward")

    print(f"Elapsed_time_s = {elapsed:.6f}")
    if figure_path is not None:
        print(f"Energy_figure = {figure_path}")


if __name__ == "__main__":
    main()
