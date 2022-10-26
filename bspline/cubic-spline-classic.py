# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Reference:
# https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html


def basic_function_base_case(knot_sequence, index, u):
    # N^0_i(u) = 1 if u_i <= u <= u_{i+1} otherwise 0
    # Reference: https://github.com/scipy/scipy/blob/v1.9.1/scipy/interpolate/_bsplines.py#L128
    if knot_sequence[index] <= u and u < knot_sequence[index + 1]:
        return 1
    else:
        return 0


def N(knot_sequence, degree_of_polynomial, index):
    if degree_of_polynomial == 0:
        return lambda u: basic_function_base_case(knot_sequence, index, u)
    else:
        N_with_lower_degree_same_index = N(
            knot_sequence, degree_of_polynomial - 1, index)
        N_with_lower_degree_higher_index = N(
            knot_sequence, degree_of_polynomial - 1, index + 1)

        def first_part(u):
            if knot_sequence[index + degree_of_polynomial] == knot_sequence[index]:
                return 0
            else:
                return (u - knot_sequence[index]) / (
                    knot_sequence[index + degree_of_polynomial] - knot_sequence[index]) * N_with_lower_degree_same_index(u)

        def second_part(u):
            if knot_sequence[index + degree_of_polynomial + 1] == knot_sequence[index + 1]:
                return 0
            else:
                return (knot_sequence[index + degree_of_polynomial + 1] - u) / (
                    knot_sequence[index + degree_of_polynomial + 1] - knot_sequence[index + 1]) * N_with_lower_degree_higher_index(u)

        return lambda u: first_part(u) + second_part(u)


def N3(knot_sequence, index):
    return N(knot_sequence, 3, index)


class CubicBSpline:
    DEGREE_OF_POLYNOMIAL = 3

    def __init__(self, knot_sequences, control_points):
        self.K = len(knot_sequences) - 1
        self.N = len(control_points)
        self.L = self.N - 1
        assert self.L == self.K - CubicBSpline.DEGREE_OF_POLYNOMIAL - \
            1, "L should be equal to K - 4"
        self.knot_sequences = knot_sequences
        self.control_points = control_points
        self.N3_i = None

    def _cox_de_boor_recursive(self, u):
        if self.N3_i is None:
            self.N3_i = [N3(self.knot_sequences, i) for i in range(self.N)]
        I = np.searchsorted(self.knot_sequences, u, 'right') - 1
        return np.sum(self.control_points[I-CubicBSpline.DEGREE_OF_POLYNOMIAL:I+1] * np.array([N3(u) for N3 in self.N3_i[I-CubicBSpline.DEGREE_OF_POLYNOMIAL:I+1]]).reshape((-1, 1)), axis=0)

    def _alpha(self, u, i, r):
        return (u - self.knot_sequences[i]) / (
            self.knot_sequences[i + 1 + CubicBSpline.DEGREE_OF_POLYNOMIAL - r] - self.knot_sequences[i])

    def _blossom_recursive(self, u, i, r):
        if r == 0:
            return self.control_points[i]
        else:
            alpha = self._alpha(u, i, r)
            return (1 - alpha) * self._blossom_recursive(u, i - 1, r - 1) + alpha * self._blossom_recursive(u, i, r - 1)

    def _de_boor_recursive(self, u):
        I = np.searchsorted(self.knot_sequences, u, 'left') - 1
        return self._blossom_recursive(u, I, CubicBSpline.DEGREE_OF_POLYNOMIAL)

    def _de_boor_dp(self, u):
        I = np.searchsorted(self.knot_sequences, u, 'left') - 1
        d = [self.control_points[j + I - CubicBSpline.DEGREE_OF_POLYNOMIAL]
             for j in range(0, CubicBSpline.DEGREE_OF_POLYNOMIAL + 1)]

        for r in range(1, CubicBSpline.DEGREE_OF_POLYNOMIAL + 1):
            for j in reversed(range(r, CubicBSpline.DEGREE_OF_POLYNOMIAL + 1)):
                i = j + I - CubicBSpline.DEGREE_OF_POLYNOMIAL
                alpha = self._alpha(u, i, r)
                d[j] = (1 - alpha) * d[j - 1] + alpha * d[j]
        return d[CubicBSpline.DEGREE_OF_POLYNOMIAL]

    def __call__(self, u):
        # using Cox-de Boor recursion formula
        # return self._cox_de_boor_recursive(u)

        # using Blossoms recursive
        # return self._de_boor_recursive(u)

        # using Blossoms dynamic programming
        return self._de_boor_dp(u)

    def plot(self):
        control_points_x, control_points_y = zip(*self.control_points)
        plt.plot(control_points_x, control_points_y)
        plt.scatter(control_points_x, control_points_y)
        # s : [u_3, u_{L+1}] \subset \mathbb{R} \mapsto \mathbb{R}^2
        curve_u = np.linspace(
            self.knot_sequences[CubicBSpline.DEGREE_OF_POLYNOMIAL], self.knot_sequences[self.N], 1000)
        curve = np.array([self(u) for u in curve_u])
        # draw curve
        curve_x, curve_y = zip(*curve)
        plt.plot(curve_x, curve_y)
        # draw knot points
        knot_points = np.array(
            [self(u) for u in self.knot_sequences[CubicBSpline.DEGREE_OF_POLYNOMIAL: self.N + 1]])
        knot_points_x, knot_points_y = zip(*knot_points)
        plt.scatter(knot_points_x, knot_points_y)

        # testing purpose
        # from scipy.interpolate import BSpline
        # spl = BSpline(self.knot_sequences, self.control_points,
        #               CubicBSpline.DEGREE_OF_POLYNOMIAL, extrapolate=False)
        # curve = spl(curve_u)
        # curve_x, curve_y = zip(*curve)
        # plt.plot(curve_x, curve_y)

def main():
    u = np.array([0, 0.0001, 0.001, 0.01, 0.5, 0.9, 0.99, 0.999, 0.9999, 1])
    d = np.array([
        [0, 0],
        [0, 2],
        [2, 3],
        [2, -1],
        [4, 0],
        [4, 2]
    ])
    
    cubic_b_splie = CubicBSpline(u, d)
    cubic_b_splie.plot()

main()