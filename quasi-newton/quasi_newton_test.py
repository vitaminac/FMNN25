import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import rosen, rosen_der, rosen_hess, fmin_bfgs
from quasi_newton import two_side_numerical_derivative, two_side_numerical_gradient, numerical_Hessian, OptimizationProblem, minimize_in_interval_with_cubic_polynomial_interpolation, minimize_in_interval_with_quadratic_polynomial_interpolation, inexact_line_serach, rosenbrock, ClassicalNewtonMethodMinimizer, ClassicalNewtonMethodWithExactLineSearchMinimizer, ClassicalNewtonMethodWithInexactLineSearchMinimizer, QuasiNewtonMethodWithSimpleBroydenRank1UpdateGoodBroydenMinimizer, QuasiNewtonMethodWithSimpleBroydenRank1UpdateBadBroydenMinimizer, QuasiNewtonMethodWithSymmetricBroydenUpdateMinimizer, QuasiNewtonMethodWithDFPRank2UpdateMinimizer, QuasiNewtonMethodWithBFGSRank2UpdateMinimizer
from chebyquad_problem import chebyquad, gradchebyquad


class TestQuasiNewton(unittest.TestCase):
    def setUp(self):
        self.f = rosen
        self.g = rosen_der
        self.G = rosen_hess
        # ensure the experiment is reproducible
        rs = np.random.RandomState(np.random.MT19937(
            np.random.SeedSequence(0)))
        self.x0 = rs.uniform(-1, +1, 2)

    def draw_evolution(self, evolution, title):
        plt.figure()
        plt.yscale("log")
        plt.plot(evolution)
        plt.title(title)
        plt.show()

    def benchmark(self, Minimizer, f=rosenbrock):
        '''
        Task 5: Test the performance of this method on the Rosenbrock function
        '''
        title = Minimizer.__name__
        optimization_problem = OptimizationProblem(f)
        minimizer = Minimizer(optimization_problem)
        X_history, y_history, x_star = minimizer.minimize(self.x0)
        # Prepare Grid Data
        x1 = x2 = np.linspace(-1, 1, 1000)
        X1, X2 = np.meshgrid(x1, x2)
        Y = f([X1, X2])
        # Draw 3D
        plt.figure()
        ax = plt.subplot(111, projection='3d',
                         xlim=(-1, +1), ylim=(-1, +1), zlim=(-100, 500))
        ax.view_init(elev=30, azim=60)
        ax.plot_surface(X1, X2, Y, color="green")
        X1_history, X2_history = zip(*X_history)
        ax.plot(X1_history, X2_history, y_history, linewidth=3.0, color="red")
        ax.set_title(title)
        # Draw Contour
        plt.figure()
        cp = plt.contourf(X1, X2, Y, levels=100)
        plt.colorbar(cp)
        plt.xlim(left=-1, right=+1)
        plt.ylim(bottom=-1, top=+1)
        plt.plot(X1_history, X2_history)
        plt.scatter(x_star[0], x_star[1])
        plt.title(title)
        self.draw_evolution(y_history, title)

    def test_rosenbrock(self):
        self.assertAlmostEqual(rosenbrock(self.x0), self.f(
            self.x0), msg="the implementation of rosenbrock is different than scipy implementation")

    def test_derivative(self):
        g = two_side_numerical_derivative(np.sin)
        self.assertAlmostEqual(np.cos(np.pi), g(
            np.pi), msg="d/dx sin(pi) should be cos(pi)")

    def test_gradient(self):
        numerical_g = two_side_numerical_gradient(self.f)
        approximate_g = numerical_g(self.x0)
        real_g = self.g(self.x0)
        g_diff = approximate_g - real_g
        self.assertTrue(np.linalg.norm(g_diff, ord=2) < 10e-8,
                        msg="the numerical gradient should approximate the real gradient")

    def test_Hessian(self):
        numerical_G = numerical_Hessian(self.g)
        approximate_G = numerical_G(self.x0)
        real_G = self.G(self.x0)
        G_diff = approximate_G - real_G
        self.assertTrue(np.linalg.norm(G_diff) < 10e-8,
                        msg="the numerical Hessian should approximate the real Hessian")

    def test_minimize_in_interval_with_cubic_polynomial_interpolatiion(self):
        pass
        self.assertAlmostEqual(0.2, minimize_in_interval_with_cubic_polynomial_interpolation(
            0.2, 1,
            0.1, 0,
            0.82, 1,
            -1.4, -2),
            msg="the implementation of minimize_in_interval_with_cubic_polynomial_interpolating is incorrect")
        self.assertAlmostEqual(0.160948, minimize_in_interval_with_cubic_polynomial_interpolation(
            0.19, 0.15,
            0.2, 0.1,
            0.8, 0.82,
            1.6, -1.4),
            places=6,
            msg="the implementation of minimize_in_interval_with_cubic_polynomial_interpolating is incorrect")
        self.assertAlmostEqual(0.160922, minimize_in_interval_with_cubic_polynomial_interpolation(
            0.181, 0.145,
            0.19, 0.1,
            0.786421, 0.82,
            1.1236, -1.4),
            places=6,
            msg="the implementation of minimize_in_interval_with_cubic_polynomial_interpolating is incorrect")

    def test_minimize_in_interval_with_quadratic_polynomial_interpolatiion(self):
        self.assertAlmostEqual(0.1, minimize_in_interval_with_quadratic_polynomial_interpolation(
            0.1, 0.5,
            0, 1,
            1, 100,
            -2),
            msg="the implementation of minimize_in_interval_with_cubic_polynomial_interpolating is incorrect")
        self.assertAlmostEqual(0.19, minimize_in_interval_with_quadratic_polynomial_interpolation(
            0.19, 0.55,
            0.1, 1,
            0.82, 100,
            -1.4),
            msg="the implementation of minimize_in_interval_with_cubic_polynomial_interpolating is incorrect")

    def test_inexact_line_search(self):
        """
        Task 7: Test this seperately from an optimization method on Rosenbrock’s function
        and use the parameters given on p.37 in the book <Practical Methods of Optimization of R. Fletcher>.
        """

        def f(alpha):
            return 100 * alpha**4 + (1 - alpha)**2

        def g(alpha):
            return 400 * alpha**3 - 2 * (1 - alpha)
        self.assertAlmostEqual(0.160948, inexact_line_serach(f, g, alpha_1=0.1),
                               places=6,
                               msg="the implementation of inexact_line_serach is incorrect")
        self.assertAlmostEqual(0.160922, inexact_line_serach(f, g, alpha_1=1),
                               places=6,
                               msg="the implementation of inexact_line_serach is incorrect")

    def test_classical_newton_method(self):
        self.benchmark(ClassicalNewtonMethodMinimizer)

    def test_newton_method_with_exact_line_search(self):
        # it could take longer to converge comparing to classfical newton method without line search, but it is more stable
        self.benchmark(ClassicalNewtonMethodWithExactLineSearchMinimizer)

    def test_newton_method_with_inexact_line_search(self):
        self.benchmark(ClassicalNewtonMethodWithInexactLineSearchMinimizer)

    def test_quasi_newton_method_with_simple_broyden_rank_1_update_good_Broyden(self):
        self.benchmark(
            QuasiNewtonMethodWithSimpleBroydenRank1UpdateGoodBroydenMinimizer)

    def test_quasi_newton_method_with_simple_broyden_rank_1_update_bad_Broyden(self):
        self.benchmark(
            QuasiNewtonMethodWithSimpleBroydenRank1UpdateBadBroydenMinimizer)

    def test_quasi_newton_method_with_symmetric_broyden_update(self):
        self.benchmark(QuasiNewtonMethodWithSymmetricBroydenUpdateMinimizer)

    def test_quasi_newton_method_with_DFP_rank_2_update(self):
        self.benchmark(QuasiNewtonMethodWithDFPRank2UpdateMinimizer)

    def test_quasi_newton_method_with_BFGS_rank_2_update(self):
        self.benchmark(QuasiNewtonMethodWithBFGSRank2UpdateMinimizer)

    def test_chebyquad(self):
        '''
        Task 10: Download from the course’s webpage the testexample chebyquad_problem.py.
        It contains the objective function chebyquad and its gradient.
        Task 11: Minimizing chebyquad corresponds to finding a set of optimal points x_j
        such that the mean value in (1) approximates best the corresponding integrals. 
        Use your code to compute these points for n = 4, 8, 11.
        Compare your results with those obtained from scipy.optimize.fmin_bfgs.
        '''
        optimization_problem = OptimizationProblem(chebyquad, gradchebyquad)
        minimizer = QuasiNewtonMethodWithBFGSRank2UpdateMinimizer(
            optimization_problem)
        # ensure the experiment is reproducible
        rs = np.random.RandomState(np.random.MT19937(
            np.random.SeedSequence(0)))
        # For n = 4
        x0 = rs.rand(4)
        X_history, y_history, x_star = minimizer.minimize(x0)
        xopt = fmin_bfgs(chebyquad, x0, gradchebyquad)
        fopt = chebyquad(xopt)
        self.draw_evolution(y_history, "chebyquad n=4")
        self.assertAlmostEqual(
            fopt,
            chebyquad(x_star),
            places=9,
            msg="our implementation of newton method doesn't work well for chebyquad n=4")
        # For n = 8
        x0 = rs.rand(8)
        X_history, y_history, x_star = minimizer.minimize(x0)
        xopt = fmin_bfgs(chebyquad, x0, gradchebyquad)
        fopt = chebyquad(xopt)
        self.draw_evolution(y_history, "chebyquad n=8")
        self.assertAlmostEqual(
            fopt,
            chebyquad(x_star),
            places=9,
            msg="our implementation of newton method doesn't work well for chebyquad n=8")
        # For n = 11
        x0 = rs.rand(11)
        X_history, y_history, x_star = minimizer.minimize(x0)
        xopt = fmin_bfgs(chebyquad, x0, gradchebyquad)
        fopt = chebyquad(xopt)
        self.draw_evolution(y_history, "chebyquad n=11")
        self.assertAlmostEqual(
            fopt,
            chebyquad(x_star),
            places=9,
            msg="our implementation of newton method doesn't work well for chebyquad n=11")

    def test_diff_BFGS_H(self):
        '''
        Task 12: The matrix H^(k) of the BFGS method should approximate G(x^(k))^{−1},
        where G is the Hessian of the problem.
        Study the quality of the approximation with growing k.
        '''
        approximated_Hessian_inverse_list = []
        real_Hessian_inverse_list = []
        G = numerical_Hessian(gradchebyquad)

        class QuasiNewtonMethodWithBFGSRank2UpdateMinimizerHessianTracker(QuasiNewtonMethodWithBFGSRank2UpdateMinimizer):
            def __init__(self, optimization_problem: OptimizationProblem):
                super().__init__(optimization_problem)

            def _update_H(self, x, new_x):
                approximated_Hessian_inverse_list.append(self._H)
                real_Hessian_inverse_list.append(np.linalg.inv(G(x)))
                super()._update_H(x, new_x)
        optimization_problem = OptimizationProblem(chebyquad, gradchebyquad)
        # ensure the experiment is reproducible
        rs = np.random.RandomState(np.random.MT19937(
            np.random.SeedSequence(0)))
        x0 = rs.rand(11)
        minimizer = QuasiNewtonMethodWithBFGSRank2UpdateMinimizerHessianTracker(
            optimization_problem)
        X_history, y_history, x_star = minimizer.minimize(x0)

        diff = [np.linalg.norm(real_Hessian_inverse - approximated_Hessian_inverse) for approximated_Hessian_inverse,
                real_Hessian_inverse in zip(approximated_Hessian_inverse_list, real_Hessian_inverse_list)]
        self.draw_evolution(diff, "Hessian inverse difference")


if __name__ == '__main__':
    unittest.main()
