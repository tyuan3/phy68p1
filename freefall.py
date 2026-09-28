import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

g = 9.8  # m/s^2
rho = 1.225  # kg/m^3


class Ball:
    # Takes mass (kg)
    def __init__(self, m, A, b=0.5, c=0.5, pos=0, dt=1 / 300) -> None:
        self.mass = m
        self.deltat = dt
        self.pos = pos
        self.vel = 0
        self.time = 0
        self.lin_drag_coeff = b
        self.quad_drag_coeff = c
        self.area = A

    def reset(self, pos=0, vel=0):
        """Reset the integration state (pos/vel/time) so the ball can be
        re-simulated from scratch. This is called before every trial run
        that curve_fit performs while it searches for the best-fit drag
        coefficient, and again before the final plotted simulation."""
        self.pos = pos
        self.vel = vel
        self.time = 0

    def _acceleration(self, vel, dragType):
        """Net acceleration (-g plus drag/mass) evaluated at an arbitrary
        velocity `vel`, rather than always at self.vel. RK4 needs this
        evaluated at several different trial velocities per step; using
        self.vel everywhere was the bug that made rk4 collapse onto
        euler_improved (see getNext)."""
        dragforce = 0
        if dragType == 1:
            dragforce = abs(self.lin_drag_coeff * vel)
        elif dragType == 2:
            dragforce = rho / 2 * vel * vel * self.area * self.quad_drag_coeff
        return -g + dragforce / self.mass

    def getNext(self, dragType=0, method='euler_improved'):
        self.time += self.deltat

        # Standard Euler method using left-endpoint approximation
        if method == 'euler':
            dvel = self._acceleration(self.vel, dragType) * self.deltat
            self.pos += self.vel * self.deltat  # Compute next position w/ left endpoint
            self.vel += dvel

        # Improved Euler method using average velocity of left and right endpoints
        elif method == 'euler_improved':
            dvel = self._acceleration(self.vel, dragType) * self.deltat
            avvel = self.vel + dvel / 2
            self.vel += dvel

            # Calculate position with average position during frame
            self.pos += avvel * self.deltat

        # Runge-Kutta 4th order method
        elif method == 'rk4':
            dt = self.deltat
            v0 = self.vel

            # Each stage's acceleration must be evaluated at that stage's
            # own trial velocity (v0, v0+k1v/2, v0+k2v/2, v0+k3v) -- not
            # all at v0 -- otherwise the four k*v terms are identical and
            # RK4 silently degenerates into euler_improved.
            k1v = self._acceleration(v0, dragType) * dt
            k1x = v0 * dt

            k2v = self._acceleration(v0 + k1v / 2, dragType) * dt
            k2x = (v0 + k1v / 2) * dt

            k3v = self._acceleration(v0 + k2v / 2, dragType) * dt
            k3x = (v0 + k2v / 2) * dt

            k4v = self._acceleration(v0 + k3v, dragType) * dt
            k4x = (v0 + k3v) * dt

            self.vel += (k1v + 2 * k2v + 2 * k3v + k4v) / 6
            self.pos += (k1x + 2 * k2x + 2 * k3x + k4x) / 6

        return (self.pos, self.vel, self.time)

    def predict(self, time, dragType=0, method='euler_improved'):
        # Get number of frames to simulate
        n = math.ceil(time / self.deltat)
        frames = []
        for _ in range(n):
            frames += [self.getNext(dragType, method)]

        return frames


def load_average_data(filename, n):
    """Load n trial files and return (t_ref, y_mean): the reference time
    grid from the first trial and the position averaged across trials.

    Different sensor setups record the raw signal with opposite sign
    conventions -- e.g. the beach-ball trials increase over time as the
    ball falls, while the foam-ball trials decrease. The simulation always
    produces a position that becomes more negative over time (falling), so
    the sign here is chosen based on each file's own raw trend rather than
    assumed fixed, so the transformed data always decreases with time
    (matching the model) regardless of which sensor convention was used.
    """
    y_all = []
    t_ref = None

    for i in range(1, n + 1):
        data = np.loadtxt(f'{filename}_{i}.txt', skiprows=1)

        t = data[:, 0]
        t -= t[0] - 0.03  # t offset
        raw_y = data[:, 1]
        delta = raw_y - raw_y[0]  # starts at 0, trends whichever way the raw sensor does

        if delta[-1] > 0:
            # Raw signal increases with time -> flip sign so the result decreases
            y = -delta + 0.002
        else:
            # Raw signal already decreases with time -> keep sign as-is
            y = delta - 0.002

        if t_ref is None:
            t_ref = t  # use first trial's time array as the reference grid

        y_all.append(y)

    y_all = np.array(y_all)          # shape (n, N)
    y_mean = y_all.mean(axis=0)      # average at each timestep

    return t_ref, y_mean


def r_squared(y_actual, y_predicted):
    y_actual = np.asarray(y_actual)
    y_predicted = np.asarray(y_predicted)
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - ss_res / ss_tot


def simulate_position_at_times(ball, t_query, coeff, dragType, method):
    """
    Model function passed to curve_fit. Runs the numerical simulation with
    a trial drag coefficient (whatever curve_fit is currently guessing) and
    returns the ball's position interpolated onto t_query, so it lines up
    with the measured data curve_fit is comparing against.
    """
    if dragType == 1:
        ball.lin_drag_coeff = coeff
    elif dragType == 2:
        ball.quad_drag_coeff = coeff

    ball.reset()
    t_end = t_query[-1]
    frames = ball.predict(t_end, dragType=dragType, method=method)

    t_sim = np.array([0.0] + [f[2] for f in frames])
    y_sim = np.array([0.0] + [f[0] for f in frames])

    y_interp = np.interp(t_query, t_sim, y_sim)

    if not np.all(np.isfinite(y_interp)):
        # A large trial coefficient (curve_fit explores these while
        # searching) combined with a light ball can make the explicit
        # integrator numerically unstable, blowing velocity/position up to
        # +-inf. Rather than letting inf/nan reach curve_fit -- which
        # crashes with "array must not contain infs or NaNs" -- return a
        # large but finite residual so the optimizer sees this as a very
        # bad fit and moves away from it instead of erroring out.
        return np.full_like(t_query, 1e6, dtype=float)

    return y_interp


def estimate_drag_guess(ball, t_data, y_data, dragType):
    """
    A single physically motivated starting guess for the drag coefficient,
    based on the ball's mass/area and the rough terminal velocity implied
    by the tail end of the measured data -- rather than a fixed absolute
    number tried regardless of the ball. A light ball (like foam) can go
    numerically unstable if curve_fit tries a coefficient sized for a much
    heavier ball, so scaling the guess by mass/area keeps the search
    starting in a physically sane range for whichever ball is passed in.

    Force balance at (roughly) terminal velocity v:
      linear drag:    m*g ~= b*v            -> b ~= m*g / v
      quadratic drag: m*g ~= 0.5*rho*c*A*v^2 -> c ~= 2*m*g / (rho*A*v^2)
    """
    v_end = (y_data[-1] - y_data[-2]) / (t_data[-1] - t_data[-2])
    v_end = max(abs(v_end), 0.1)  # guard against a near-zero velocity estimate

    if dragType == 1:
        return ball.mass * g / v_end
    else:
        return 2 * ball.mass * g / (rho * ball.area * v_end ** 2)


def fit_drag_coefficient(ball, t_data, y_data, dragType, method,
                          guess_factors=(0.1, 0.3, 1, 3, 10)):
    """
    Least-squares fit of the drag coefficient (scipy.optimize.curve_fit is
    the Python equivalent of MATLAB's lsqcurvefit). Because the "model" is
    a numerical integration rather than a closed-form curve, the fit is
    tried from several starting guesses -- scaled around a physically
    motivated base estimate (see estimate_drag_guess) rather than fixed
    absolute values -- and the run with the lowest sum-of-squared-residuals
    is kept. This is the "redo the fit until it's minimized" step.
    """
    base_guess = estimate_drag_guess(ball, t_data, y_data, dragType)
    initial_guesses = [f * base_guess for f in guess_factors]

    best = None
    for p0 in initial_guesses:
        try:
            popt, _ = curve_fit(
                lambda t, coeff: simulate_position_at_times(ball, t, coeff, dragType, method),
                t_data, y_data, p0=[p0], bounds=(0, np.inf), method='trf', maxfev=5000
            )
        except (RuntimeError, ValueError):
            continue

        coeff = popt[0]
        y_pred = simulate_position_at_times(ball, t_data, coeff, dragType, method)
        ss_res = np.sum((y_data - y_pred) ** 2)

        if best is None or ss_res < best[1]:
            best = (coeff, ss_res, y_pred)

    if best is None:
        raise RuntimeError(f"Fit failed to converge for dragType={dragType}, method={method}")

    coeff, ss_res, y_pred = best
    r2 = r_squared(y_data, y_pred)
    return coeff, r2


DRAG_LABELS = {1: "Linear", 2: "Quadratic"}
METHOD_LABELS = {
    'euler': 'Standard Euler',
    'rk4': 'RK4',
}
METHOD_COLORS = {
    'euler': 'tab:blue',
    'rk4': 'tab:orange',
}


def run_fit_and_plot(ball, balltype, t_data, y_data, t_fit, y_fit, methods, filename_suffix='', title_suffix=''):
    """
    Fit each method's drag coefficient using only (t_fit, y_fit), then plot
    the resulting prediction curves against the FULL (t_data, y_data) so you
    can see how well a fit generalizes beyond the data it was fit on.
    filename_suffix keeps separate fits (e.g. full data vs. first-half-only)
    from overwriting each other's saved figures.
    """
    t_end = t_data[-1]
    fit_cutoff = t_fit[-1]
    fit_is_partial = fit_cutoff < t_end

    for dragType in (1, 2):
        coeff_name = 'b' if dragType == 1 else 'c'

        plt.figure(figsize=(9, 6))
        plt.errorbar(t_data, y_data, yerr=0.02, fmt='o', color='dimgray', ecolor='dimgray', markersize=4,
                     capsize=3, zorder=10, label='Measured data (avg. of 5 trials)')

        if fit_is_partial:
            # Mark where the fitting data stopped
            plt.axvline(fit_cutoff, color='gray', linestyle=':', linewidth=1.5, zorder=5,
                        label=f'End of fit window (t={fit_cutoff:.3f} s)')

        for method in methods:
            coeff, r2 = fit_drag_coefficient(ball, t_fit, y_fit, dragType, method)

            # Re-simulate at full frame resolution over the full time range
            if dragType == 1:
                ball.lin_drag_coeff = coeff
            else:
                ball.quad_drag_coeff = coeff
            ball.reset()
            frames = ball.predict(t_end, dragType=dragType, method=method)
            t_sim = np.array([0.0] + [f[2] for f in frames])
            y_sim = np.array([0.0] + [f[0] for f in frames])

            color = METHOD_COLORS[method]
            label = (f"{METHOD_LABELS[method]}: "
                     f"{coeff_name}={coeff:.4f}, $R^2$={r2:.4f}")
            plt.plot(t_sim, y_sim, '-', color=color, linewidth=2, label=label)

        plt.title(f'{balltype.capitalize()} Freefall with {DRAG_LABELS[dragType]} Drag '
            f'(dt = {ball.deltat:.4g} s){title_suffix}')
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.grid(True)
        plt.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.savefig(f'Figures/{balltype}_{DRAG_LABELS[dragType].lower()}{filename_suffix}', dpi=300)


def main():
    balls = [Ball(m=0.00485, A=0.00477663773),
             Ball(m=0.075, A=0.002299788928),
             Ball(m=0.0372, A=0.04171260073)]

    ball = balls[1]  # Manual hardcoded ball selection (change for different balls)

    balltype = 'rubber'  # Remember to hard code this for plot labels to be correct balls[0] is 'foam', balls[1] is 'rubber', balls[2] is 'beach'
    t_data, y_data = load_average_data(f'data/{balltype}/{balltype.capitalize()}', 5)

    methods = ['euler', 'rk4']

    # Pass 1: fit using the full dataset (original behavior).
    run_fit_and_plot(ball, balltype, t_data, y_data, t_data, y_data, methods)

    # Pass 2: fit using only the first half of the data, saved under a
    # different filename suffix so it doesn't overwrite the full-data fit.
    half = len(t_data) // 2
    t_fit_half = t_data[:half]
    y_fit_half = y_data[:half]
    run_fit_and_plot(ball, balltype, t_data, y_data, t_fit_half, y_fit_half, methods,
                      filename_suffix='_firsthalf_fit', title_suffix=' (fit to first half of data)')

    plt.show()


if (__name__ == "__main__"):
    main()