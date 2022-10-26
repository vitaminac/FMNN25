import numpy as np


def rosenbrock(x):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen.html
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# reference:
# https://docs.scipy.org/doc/scipy/reference/optimize.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BFGS.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.SR1.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.HessianUpdateStrategy.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html


# https://en.wikipedia.org/wiki/Numerical_differentiation#Step_size
MACHINE_EPSILON_FOR_DOUBLE_PRECISION = np.finfo(float).eps
# For basic central differences, the optimal step is the cube-root of machine epsilon.
NUMERICAL_STABLE_STEP_SIZE_FOR_DOUBLE_PRECISION = np.cbrt(
    MACHINE_EPSILON_FOR_DOUBLE_PRECISION)


# https://en.wikipedia.org/wiki/Numerical_differentiation
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.approx_fprime.html
def two_side_numerical_derivative(f, h=NUMERICAL_STABLE_STEP_SIZE_FOR_DOUBLE_PRECISION):
    def g(x):
        return (f(x + h) - f(x - h)) / (2 * h)
    return g


def two_side_numerical_gradient(f, h=NUMERICAL_STABLE_STEP_SIZE_FOR_DOUBLE_PRECISION):
    def g(x):
        X_left = np.vstack([x for i in range(x.size)])
        X_right = np.vstack([x for i in range(x.size)])
        for i in range(x.size):
            X_left[i, i] -= h
            X_right[i, i] += h
        y_left = np.apply_along_axis(f, axis=1, arr=X_left)
        y_right = np.apply_along_axis(f, axis=1, arr=X_right)
        return (y_right - y_left) / (2 * h)
    return g


def numerical_Hessian(g, h=NUMERICAL_STABLE_STEP_SIZE_FOR_DOUBLE_PRECISION):
    def H(x):
        """
        Approximate the Hessian by finite differences and a symmetrizing step.
        G := \frac{1}{2} * (\overline{X} + \overline{X}^{T})
        """
        overline_G = np.zeros((x.size, x.size))
        for i in range(x.size):
            epsilon = np.zeros(x.shape)
            epsilon[i] += h
            x_left = x - epsilon
            x_right = x + epsilon
            prime_left = g(x_left)
            prime_right = g(x_right)
            overline_G[i, :] = (prime_right - prime_left) / (2 * h)
        return 0.5 * (overline_G + overline_G.T)
    return H


class OptimizationProblem:
    '''
    Task 1: design an optimization problem class
            which take an objective function as input and optionally its gradient
    '''

    def __init__(self, f, g=None, G=None):
        assert callable(f), "f must be a function"
        self._f = f
        self._g = g if g is not None else two_side_numerical_gradient(f)
        self._G = G if G is not None else numerical_Hessian(self._g)

    def f(self, x):
        return self._f(x)

    def g(self, x):
        return self._g(x)

    def G(self, x):
        return self._G(x)


def minimize_in_interval_with_cubic_polynomial_interpolation(
        left, right,
        a, b,
        f_a, f_b,
        f_prime_a, f_prime_b):
    """
    <Practical Methods of Optimization of R. Fletcher>
    p. 37 Hermite interpolating cubic polynomial
    """
    # Mapping [a, b] on to [0, 1] in z-space and a > b is allowed
    if a > b:
        a, b = b, a
        f_a, f_b = f_b, f_a
        f_prime_a, f_prime_b = f_prime_b, f_prime_a
    # remap the interval values
    interval_left = (left - a) / (b - a)
    interval_right = (right - a) / (b - a)
    if interval_left > interval_right:
        interval_left, interval_right = interval_right, interval_left
    # remap the f values
    f_0 = f_a
    f_1 = f_b
    # remap the slope values
    f_prime_0 = f_prime_a * (b - a)
    f_prime_1 = f_prime_b * (b - a)
    # \eta = 3 * (f_1 - f_0) - 2 * f'_0 - f'_1
    eta = 3 * (f_1 - f_0) - 2 * f_prime_0 - f_prime_1
    # \xi = f'_0 + f'_1 - 2 * (f_1 - f_0)
    xi = f_prime_0 + f_prime_1 - 2 * (f_1 - f_0)
    # c(z) = f_0 + f'_0 * z + \eta * z^2 + \xi * z^3
    polynomial = np.poly1d([xi, eta, f_prime_0, f_0])
    # To find the minimizers of c(z) in a given interval,
    # it is necessary to examine not only the stationary values,
    # but also the values and derivatives at the ends of the interval.
    # This requires some simple calculus.
    stationary_point = None

    # cubic polynomial
    if polynomial.order == 3:
        # To find the x coordinate of stationary point of a cubic polynomial a*x^3 + b*x^3 + c*x + d
        # is the same as finding the critical points,
        # which is find the roots of derivarive of cubic polynomial
        # which is quadric polynomial 3*a*x + 2*b*x + c =0
        # The solution to this equation is \frac{-b +/- \sqrt{b^2 - 3ac}}{3a}
        # https://en.wikipedia.org/wiki/Cubic_function#Critical_and_inflection_points
        a_coeficient = polynomial[3]
        b_coeficient = polynomial[2]
        c_coeficient = polynomial[1]
        # the sign of the expression in side the square determines the number of critical points
        value_inside_of_squart = b_coeficient**2 - 3 * a_coeficient * c_coeficient
        # if it is positive then there are two critical points,
        # one is local maximum and the other is a local minimum
        if value_inside_of_squart > 0:
            choice_1 = (-b_coeficient +
                        np.sqrt(value_inside_of_squart)) / (3 * a_coeficient)
            choice_2 = (-b_coeficient -
                        np.sqrt(value_inside_of_squart)) / (3 * a_coeficient)
            if polynomial(choice_1) < polynomial(choice_2):
                # local minumum is choice_1
                stationary_point = choice_1
            else:
                # local minumum is choice_2
                stationary_point = choice_2
        # if the value is zero then there is only one critical point
        elif value_inside_of_squart == 0:
            stationary_point = -b_coeficient / (3 * a)
        # if value is negative, the cubic function is strictly monotonic
        else:
            pass
    # quadratic polynomial
    elif polynomial.order == 2:
        # https://en.wikipedia.org/wiki/Quadratic_equation#Geometric_interpretation
        # the formula to find the x coordinate of stationary point of a quadratic polynomial a*x^2 + b*x + c is x = -b / (2 * a)
        a_coeficient = polynomial[2]
        b_coeficient = polynomial[1]
        stationary_point = -b_coeficient / (2 * a_coeficient)
    else:
        # for linear function the best function can only occurs at extremes
        pass
    if stationary_point is not None and interval_left < stationary_point < interval_right:
        possible_z_values = [interval_left, stationary_point, interval_right]
    else:
        possible_z_values = [interval_left, interval_right]
    best_z = possible_z_values[np.argmin(polynomial(possible_z_values))]
    # transform z back to alpha
    best_alpha = a + best_z * (b - a)
    return best_alpha


def minimize_in_interval_with_quadratic_polynomial_interpolation(
        left, right,
        a, b,
        f_a, f_b,
        f_prime_a):
    """
    <Practical Methods of Optimization of R. Fletcher>
    p. 37 quadratic interpolation
    """
    f_prime_0 = f_prime_a * (b - a)
    polynomial = np.poly1d([f_b - f_a - f_prime_0, f_prime_0, f_a])
    # remap the interval values
    interval_left = (left - a) / (b - a)
    interval_right = (right - a) / (b - a)
    if interval_left > interval_right:
        interval_left, interval_right = interval_right, interval_left
    stationary_point = None
    if polynomial.order == 2:
        # https://en.wikipedia.org/wiki/Quadratic_equation#Geometric_interpretation
        # the formula to find the x coordinate of stationary point of a quadratic polynomial a*x^2 + b*x + c is x = -b / (2 * a)
        a_coeficient = polynomial[2]
        b_coeficient = polynomial[1]
        stationary_point = -b_coeficient / (2 * a_coeficient)
    if stationary_point is not None and interval_left < stationary_point < interval_right:
        possible_z_values = [interval_left, stationary_point, interval_right]
    else:
        possible_z_values = [interval_left, interval_right]
    best_z = possible_z_values[np.argmin(polynomial(possible_z_values))]
    # transform z back to alpha
    best_alpha = a + best_z * (b - a)
    return best_alpha


def inexact_line_serach(f, g, alpha_1=None, goldstein_rho=0.01, sigma=0.1, tau_1=9.0, tau_2=0.1, tau_3=0.5, low_bound=0.0):
    """
    Task 6: Write an inexact line search method
    based on the Goldstein/Wolfe conditions

    <Practical Methods of Optimization of R. Fletcher>
    page 27, equations 2.5.1, 2.5.2
    page 29, equation 2.5.6
    page 34 algorithm 2.6.2
    page 35 algorithm 2.6.4
    """
    assert f is not None, "f(alpha) = f(x^k + \alpha * s^k) must be present"
    assert low_bound is not None, "To ensure f(\alpha) intersects the p-line, we need supply a lower bound \overfline{f}. In a nonlinear least squares problem \overline{f} = 0 would be appropriate."
    assert 0 < goldstein_rho < 0.5, "Goldstein rho should be a fixed parameter \in (0, 1/2)"
    assert goldstein_rho < sigma < 1, "f'(\alpha) >= \sigma f'(0)"
    assert 0 < tau_1, "\tau_1 is a preset factor by which the size of the jumps is increased, typically \tau_1 = 9"
    assert 0 < tau_2 < tau_3 <= 0.5, "\tau_2 and \tau_3 are preset factors 0 < \tau_2 < \tau_3 <= \frac{1}{2} which restrict \alpha{j} from being arbitrarily close to the extremes of the interval [a_j, b_j]. It then follows that | b_{j+1} - a_{j+1} | <= (1 - \tau_2) * | b_j - a_j | and this guarantees convergence of the inverval lengths to zero. Typical values are \tau_2 = \frac{1}{10} and \tau_3 = {1}{2}"
    f0 = f(0)
    assert g is not None, "we assumed the first derivative is available"
    g0 = g(0)
    assert g0 < 0, "we assumed the descent condition f'(0) < 0 is hold"
    # \mu = (\overline{f} - f(0)) / \rho * f'(0)
    mu = (low_bound - f0) / (goldstein_rho * g0)

    # The bracketing phase which searches to find a bracket
    # that is a non-trivial interval [a, b]
    # which is known to cintain an interval of acceptable points
    # In the bracketing phase the iterates \alpha_i move out to the right
    # in increasingly large jumps until either f <= \overline{f} is detected
    # or a bracket on an interval of acceptable points is located.
    previous_alpha = 0  # Intially \alpha_0 = 0
    f_previous_alpha = None
    f_prime_previous_alpha = None
    assert alpha_1 is None or 0 < alpha_1 <= mu, "\alpha_1 is given (0 < \alpha_1 <= \mu)"
    alpha = alpha_1 if alpha_1 is not None else mu / 2
    f_alpha = None
    f_prime_alpha = None
    a = None
    b = None
    f_a = None
    f_b = None
    f_prime_a = None
    f_prime_b = None
    while True:
        # evualua f(\alpha_i)
        f_alpha = f(alpha)
        # if f(\alpha_i) <= \overline{f} then terminate
        if f_alpha <= low_bound:
            return alpha
        # armijo rule (equation 2.5.1)
        # the formula in the book is wrong is should be follwing:
        # if f(\alpha_i) > f(0) + \alpha_i * \rho * f'(0) or f(\alpha_i) >= f(\alpha_{i-1})
        f_previous_alpha = f(previous_alpha)
        if f_alpha > f0 + alpha * goldstein_rho * g0 or f_alpha >= f_previous_alpha:
            # a_i = \alpha_{i-1}
            a = previous_alpha
            f_a = f_previous_alpha
            f_prime_a = f_prime_previous_alpha
            # b_i = \alpha_i
            b = alpha
            f_b = f_alpha
            f_prime_b = f_prime_alpha
            # terminate B end
            break
        f_prime_alpha = g(alpha)
        if np.abs(f_prime_alpha) < - sigma * g0:
            return alpha
        if f_prime_alpha >= 0:
            a = alpha
            f_a = f_alpha
            b = previous_alpha
            f_b = f_previous_alpha
            f_prime_a = f_prime_alpha
            f_prime_b = f_prime_previous_alpha
            break
        if mu <= 2 * alpha - previous_alpha:
            previous_alpha = alpha
            alpha = mu
            f_previous_alpha = f_alpha
            f_prime_previous_alpha = f_prime_alpha
            f_alpha = None
            f_prime_alpha = None
        else:
            if f_prime_previous_alpha is None:
                f_prime_previous_alpha = g(previous_alpha)
            left = 2 * alpha - previous_alpha
            right = np.min([mu, alpha + tau_1 * (alpha - previous_alpha)])
            new_alpha = minimize_in_interval_with_cubic_polynomial_interpolation(
                left, right,
                alpha, previous_alpha,
                f_alpha, f_previous_alpha,
                f_prime_alpha, f_prime_previous_alpha)
            previous_alpha = alpha
            alpha = new_alpha
            f_previous_alpha = f_alpha
            f_alpha = None
            f_prime_previous_alpha = f_prime_alpha
            f_prime_alpha = None
    # Sectioning phase in which the bracket
    # is sectioned to generate a sequence of brackets [a_j, b_j]
    # whose length tends to zero
    assert a is not None
    assert b is not None
    while True:
        # choose \alpha_j in [\alpha_j + \tau_2 \cdot (b_j - a_j), b_j - \tau_3 \cdot (b_j - a_j)]
        left = a + tau_2 * (b - a)
        right = b - tau_3 * (b - a)
        if f_a is None:
            f_a = f(a)
        if f_b is None:
            f_b = f(b)
        if f_prime_a is None:
            f_prime_a = g(a)
        if f_prime_b is not None:
            alpha = minimize_in_interval_with_cubic_polynomial_interpolation(
                left, right,
                a, b,
                f_a, f_b,
                f_prime_a, f_prime_b)
        else:
            alpha = minimize_in_interval_with_quadratic_polynomial_interpolation(
                left, right,
                a, b,
                f_a, f_b,
                f_prime_a)
        f_alpha = f(alpha)
        f_prime_alpha = None
        # Armijo Rule
        if f_alpha > f0 + goldstein_rho * alpha * g0 or f(alpha) >= f(a):
            b = alpha
            f_b = f_alpha
            f_prime_b = f_prime_alpha
        else:
            f_prime_alpha = g(alpha)
            if np.abs(f_prime_alpha) <= -sigma * g0:
                return alpha
            else:
                if (b - a) * f_prime_alpha >= 0:
                    b = a
                    f_b = f_a
                    f_prime_b = f_prime_a
                # interchange line of a_{j+1} := a_j with
                # if (b_j - a_j)f'(a_j) >= 0 then b_{j+1} := a_{j} else b_{j+1} := b_j
                a = alpha
                f_a = f_alpha
                f_prime_a = f_prime_alpha


class NewtonMethodMinimizer:
    _MAX_ITERATIONS = 10e4
    _RESIDUAL_CRITERION_THRESHOLD = NUMERICAL_STABLE_STEP_SIZE_FOR_DOUBLE_PRECISION
    _CAUCHY_CRITERION_THRESHOLD = NUMERICAL_STABLE_STEP_SIZE_FOR_DOUBLE_PRECISION

    def __init__(self, optimization_problem: OptimizationProblem, line_search_method: str = None):
        self._optimization_problem = optimization_problem
        self._line_search_method = line_search_method

    def minimize(self, x0):
        assert np.isscalar(
            x0) or x0.ndim == 1, "x_0 should be a scalar or vector"
        X_history = []
        y_history = []
        k = 0
        x = x0
        new_x = self._step_toward_descent_direction(x)
        while not self._meet_termination_criterio(k, x, new_x):
            X_history.append(x)
            y_history.append(self._optimization_problem.f(x))
            x = new_x
            new_x = self._step_toward_descent_direction(x)
            k += 1
        x_star = x
        X_history.append(x_star)
        y_history.append(self._optimization_problem.f(x_star))
        return X_history, y_history, x_star

    def _step_toward_descent_direction(self, x):
        newton_direction = self._newton_direction(x)
        new_x = x + self._step_size(x, newton_direction) * newton_direction
        return new_x

    def _newton_direction(self, x):
        raise NotImplementedError()

    def _step_size(self, x, newton_direction):
        if self._line_search_method is None:
            return 1
        else:
            return self._line_search(x, newton_direction)

    def _line_search(self, x, newton_direction):
        def phi(alpha):
            return self._optimization_problem.f(x + alpha * newton_direction)

        if self._line_search_method == "exact":
            alpha = np.logspace(-16, 3, 2000)
            v_phi = np.vectorize(phi)
            y = v_phi(alpha)
            # scipy.optimize.minimize(phi, 0).x
            best_alpha = alpha[np.argmin(y)]
            return best_alpha
        elif self._line_search_method == "inexact":
            phi_g = two_side_numerical_derivative(phi)
            return inexact_line_serach(phi, phi_g)
        else:
            raise NotImplementedError()

    def _meet_termination_criterio(self, k, x, new_x):
        return self._meet_residual_criterion(new_x) or self._meet_cauchy_criterion(x, new_x) or self._has_convergence_problem(k, new_x)

    def _meet_residual_criterion(self, x):
        """
        The Newton iteration is stopped as soon as the residual || g(x^{k}) || is small enough.
        """
        return np.linalg.norm(self._optimization_problem.g(x)) <= NewtonMethodMinimizer._RESIDUAL_CRITERION_THRESHOLD

    def _meet_cauchy_criterion(self, x, new_x):
        """
        Terminate the iteration as soon as the Newton-correction
        || x^{k+1} − x^{k} || is small enough.
        """
        return np.linalg.norm(new_x - x, ord=2) <= NewtonMethodMinimizer._CAUCHY_CRITERION_THRESHOLD

    def _has_convergence_problem(self, k, new_x):
        return k >= NewtonMethodMinimizer._MAX_ITERATIONS or np.any(np.isinf(new_x)) or np.any(np.isnan(new_x))


class ClassicalNewtonMethodMinimizer(NewtonMethodMinimizer):
    '''
    Task 3: Implement classical Newton’s method
            for finding a minimum of the objective function.
    '''

    def __init__(self, optimization_problem: OptimizationProblem, line_search_method: str = None):
        super().__init__(optimization_problem, line_search_method)

    def _newton_direction(self, x):
        return -(np.linalg.inv(self._optimization_problem.G(x)) @ self._optimization_problem.g(x))


class ClassicalNewtonMethodWithExactLineSearchMinimizer(ClassicalNewtonMethodMinimizer):
    '''
    Task 4: Provide Newton’s method with exact line search method
    '''

    def __init__(self, optimization_problem: OptimizationProblem):
        super().__init__(optimization_problem, "exact")


class ClassicalNewtonMethodWithInexactLineSearchMinimizer(ClassicalNewtonMethodMinimizer):
    '''
    Task 8: Provide Newton’s method with inexact line search method
    '''

    def __init__(self, optimization_problem: OptimizationProblem):
        super().__init__(optimization_problem, "inexact")


class QuasiNewtonMethodMinimizer(NewtonMethodMinimizer):
    '''
    Task 2: Design a general optimization method class.
            It should be generic for special kinds of Quasi-Newton methods.
    '''

    def __init__(self, optimization_problem: OptimizationProblem, line_search_method: str = "inexact"):
        super().__init__(optimization_problem, line_search_method)
        self._H = None

    def minimize(self, x0):
        self._H = np.eye(x0.size)
        return super().minimize(x0)

    def _step_toward_descent_direction(self, x):
        new_x = super()._step_toward_descent_direction(x)
        self._update_H(x, new_x)
        return new_x

    def _newton_direction(self, x):
        return -(self._H @ self._optimization_problem._g(x))

    def _update_H(self, x):
        raise NotImplementedError()


'''
Task 9: Derive from Newton’s method five classes of Quasi Newton methods:

    * Simple Broyden rank-1 update of G and applying Sherman-Morisson’s formula ("good Broyden")
    * Simple Broyden rank-1 update of H = G−1 ("bad Broyden")
    * Symmetric Broyden update
    * DFP rank-2 update
    * BFGS rank-2 update

'''


class QuasiNewtonMethodWithSimpleBroydenRank1UpdateGoodBroydenMinimizer(QuasiNewtonMethodMinimizer):
    def __init__(self, optimization_problem: OptimizationProblem):
        # we have to use exact line search because H is not always positive definite
        super().__init__(optimization_problem, "exact")

    def _update_H(self, x, new_x):
        # Given Q^k, find Q^{k+1} by solving min | Q^{k+1} - Q^k |_F
        # subject to Q^{k+1} \delta^k = \gamma^k
        # where \delta^k is x^{k+1} - x^k and \gamma^k is g^{k+1} - g^k
        delta = new_x - x
        gamma = self._optimization_problem.g(
            new_x) - self._optimization_problem.g(x)
        # Sherman – Morrison formula
        # H^{k+1} = H^k + \frac{\delta^{k} - H^k \gamma^k}{\delta^k^T H^k \gamma^k} \delta^k^T H^k
        H_dot_gamma = self._H @ gamma
        numerator = delta - H_dot_gamma
        denominator = delta @ H_dot_gamma
        fraction = numerator / denominator
        self._H = self._H + np.outer(fraction, delta) * self._H


class QuasiNewtonMethodWithSimpleBroydenRank1UpdateBadBroydenMinimizer(QuasiNewtonMethodMinimizer):
    def __init__(self, optimization_problem: OptimizationProblem):
        # we have to use exact line search because H is not always positive definite
        super().__init__(optimization_problem, "exact")

    def _update_H(self, x, new_x):
        delta = new_x - x
        gamma = self._optimization_problem.g(
            new_x) - self._optimization_problem.g(x)
        # H^{k+1} = H^k + \frac{\delta^{k} - H^k \gamma^k}{\gamma^k^T \gamma^k} \gamma^k^T
        H_dot_gamma = self._H @ gamma
        numerator = delta - H_dot_gamma
        denominator = np.dot(gamma, gamma)
        fraction = numerator / denominator
        self._H = self._H + np.outer(fraction, gamma)


class QuasiNewtonMethodWithSymmetricBroydenUpdateMinimizer(QuasiNewtonMethodMinimizer):
    def __init__(self, optimization_problem: OptimizationProblem):
        # we have to use exact line search because H is not always positive definite
        super().__init__(optimization_problem, "exact")

    def _update_H(self, x, new_x):
        delta = new_x - x
        gamma = self._optimization_problem.g(
            new_x) - self._optimization_problem.g(x)
        u = delta - np.dot(self._H, gamma)
        a = 1 / np.dot(u, gamma)
        self._H = self._H + a * np.outer(u, u)


class QuasiNewtonMethodWithDFPRank2UpdateMinimizer(QuasiNewtonMethodMinimizer):
    def __init__(self, optimization_problem: OptimizationProblem):
        super().__init__(optimization_problem)

    def _update_H(self, x, new_x):
        delta = new_x - x
        gamma = self._optimization_problem.g(
            new_x) - self._optimization_problem.g(x)
        delta_dot_gamma = np.dot(delta, gamma)
        H_dot_gamma = np.dot(self._H, gamma)
        second_part = np.outer(delta, delta) / delta_dot_gamma
        last_part = np.dot(np.outer(H_dot_gamma, gamma),
                           self._H) / np.dot(gamma, H_dot_gamma)
        self._H = self._H + second_part - last_part


class QuasiNewtonMethodWithBFGSRank2UpdateMinimizer(QuasiNewtonMethodMinimizer):
    def __init__(self, optimization_problem: OptimizationProblem):
        super().__init__(optimization_problem)

    def _update_H(self, x, new_x):
        delta = new_x - x
        gamma = self._optimization_problem.g(
            new_x) - self._optimization_problem.g(x)
        delta_dot_gamma = np.dot(delta, gamma)
        H_dot_gamma = np.dot(self._H, gamma)
        second_part_factor = 1 + np.dot(gamma, H_dot_gamma) / delta_dot_gamma
        second_part_fraction = np.outer(delta, delta) / delta_dot_gamma
        second_part = second_part_factor * second_part_fraction
        last_part_numerator = (
            np.outer(H_dot_gamma, delta) + np.dot(np.outer(delta, gamma), self._H))
        last_part = last_part_numerator / delta_dot_gamma
        self._H = self._H + second_part - last_part
