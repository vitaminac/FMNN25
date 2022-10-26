# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded


def basic_function_base_case(knot_sequence, index, u):
    # N^0_i(u) = 1 if u_{i-1} <= u < u_i otherwise 0
    if knot_sequence[index - 1] <= u and u < knot_sequence[index]:
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
            if knot_sequence[index + degree_of_polynomial - 1] == knot_sequence[index - 1]:
                # we use the convention 0/0 = 0 if nodes coincide
                return 0
            else:
                return (u - knot_sequence[index - 1]) / (
                    knot_sequence[index + degree_of_polynomial - 1] - knot_sequence[index - 1]) * N_with_lower_degree_same_index(u)

        def second_part(u):
            if index + degree_of_polynomial > len(knot_sequence) - 1:
                # The recursion seems to require grid points u_{−1} and u_{K+1}
                # but the location of these points does not affect the final result
                # for u \in [u_{2}, u_{K−2}] in case of cubic splines
                # as related terms will be multiplied by zero.
                return 0
            elif knot_sequence[index + degree_of_polynomial] == knot_sequence[index]:
                # we use the convention 0/0 = 0 if nodes coincide
                return 0
            else:
                return (knot_sequence[index + degree_of_polynomial] - u) / (
                    knot_sequence[index + degree_of_polynomial] - knot_sequence[index]) * N_with_lower_degree_higher_index(u)

        def f(u):
            return first_part(u) + second_part(u)
        return f


def N3(knot_sequence, index):
    return N(knot_sequence, 3, index)


class CubicBSpline:
    DEGREE_OF_POLYNOMIAL = 3

    def __init__(self, knot_sequences, control_points):
        self.K = len(knot_sequences) - 1
        self.L = len(control_points) - 1
        assert self.L == self.K - 2, "L should be equal to K - 2"
        self.knot_sequences = knot_sequences
        self.control_points = control_points
        self.N3_i = None

    def _cox_de_boor_recursive(self, u):
        if self.N3_i is None:
            self.N3_i = [N3(self.knot_sequences, i) for i in range(self.L + 1)]
        I = np.searchsorted(self.knot_sequences, u, 'right') - 1
        # ... u_{I−2} <= u_{I−1} <= u_{I} ≤ u < u_{I+1} <= u_{I+2} <= u_{I+3} ...
        # s(u) = \sum_{i=I-2}^{I+1} d_{i} * N^3_i(u)
        return np.sum(self.control_points[I-2:I+2] * np.array([f(u) for f in self.N3_i[I-2:I+2]]).reshape((-1, 1)), axis=0)

    def _alpha(self, u, I, column):
        leftmost_index = I - 1
        rightmost_index = I + CubicBSpline.DEGREE_OF_POLYNOMIAL - column
        return (self.knot_sequences[rightmost_index] - u) / (self.knot_sequences[rightmost_index] - self.knot_sequences[leftmost_index])

    def _blossom_recursive(self, u, I, column):
        if column == 0:
            return self.control_points[I]
        else:
            alpha = self._alpha(u, I, column)
            return alpha * self._blossom_recursive(u, I - 1, column - 1) + (1 - alpha) * self._blossom_recursive(u, I, column - 1)

    def _de_boor_recursive(self, u):
        # Situation: u \in [u_{I}, u_{I+1}]:
        I = np.searchsorted(self.knot_sequences, u, 'left') - 1
        return self._blossom_recursive(u, I + 1, CubicBSpline.DEGREE_OF_POLYNOMIAL)

    def __call__(self, u):
        # using Cox-de Boor recursion formula
        # return self._cox_de_boor_recursive(u)
        # using Blossoms recursive
        return self._de_boor_recursive(u)

    def plot(self):
        control_points_x, control_points_y = zip(*self.control_points)
        plt.plot(control_points_x, control_points_y)
        plt.scatter(control_points_x, control_points_y)
        # s : [u_2, u_K-2] \subset \mathbb{R} \mapsto \mathbb{R}^2
        curve_u = np.linspace(
            self.knot_sequences[2], self.knot_sequences[self.L], 1000)
        curve = np.array([self(u) for u in curve_u])
        # draw curve
        curve_x, curve_y = zip(*curve)
        plt.plot(curve_x, curve_y)
        # draw knot points
        knot_points = np.array(
            [self(u) for u in self.knot_sequences[2: self.L + 1]])
        knot_points_x, knot_points_y = zip(*knot_points)
        plt.scatter(knot_points_x, knot_points_y)

    @staticmethod
    def _print_matrix(a):
        N = len(a)
        M = len(a[0])
        for i in range(N):
            for j in range(M):
                print(str(round(a[i][j], 5)).ljust(10), sep=' ', end=' ')
            print()

    @staticmethod
    def _convert_to_diagonal_form(a):
        N = len(a)
        M = len(a[0])
        # CubicBSpline._print_matrix(a)
        assert N == M, "should be square matrix"
        I = CubicBSpline.DEGREE_OF_POLYNOMIAL - 1
        u = CubicBSpline.DEGREE_OF_POLYNOMIAL - 1
        ab = np.zeros((I + u + 1, M))
        for i in range(N):
            for j in range(M):
                if u + i - j < I + u + 1:
                    ab[u + i - j][j] = a[i, j]
        # print("=====")
        # CubicBSpline._print_matrix(ab)
        return (ab, I, u)

    @staticmethod
    def interpolate(knot_sequences, data_points):
        K = len(knot_sequences) - 1
        L = len(data_points) - 1
        assert L == K - 2, "L should be equal to K - 2"
        # To be able to evaluate this system,
        # the first and last three grid points
        # need to have multiplicity three:
        # u_0 = u_1 = u_2 and u_{K−2} = u_{K−1} = u_K
        assert knot_sequences[0] == knot_sequences[1] and knot_sequences[1] == knot_sequences[
            2], "the first three grid points need to have multiplicity three u_0 = u_1 = u_2"
        assert knot_sequences[K-2] == knot_sequences[K-1] and knot_sequences[K-1] == knot_sequences[
            K], "the last three grid points need to have multiplicity three u_{K−2} = u_{K−1} = u_K"
        # calculate left hand side matrix
        a = np.zeros((L + 1, L + 1))
        N3_i = [N3(knot_sequences, i) for i in range(L + 1)]
        for i in range(L + 1):
            for j in range(L + 1):
                greville_abscissae = (
                    knot_sequences[i] + knot_sequences[i + 1] + knot_sequences[i + 2]) / 3
                a[i][j] = N3_i[j](greville_abscissae)
        # the last control point should coincide with last datapoint
        # we need to manually assign the coefficient = 1.0
        # because u_{L} is outside of domain of N3_L
        a[L][L] = 1.0
        ab, I, u = CubicBSpline._convert_to_diagonal_form(a)
        rhs_x = data_points[:, 0]
        rhs_y = data_points[:, 1]
        control_points_x = solve_banded((I, u), ab, rhs_x)
        control_points_y = solve_banded((I, u), ab, rhs_y)
        control_points = np.stack((control_points_x, control_points_y), axis=1)
        assert len(control_points) == len(data_points)
        return CubicBSpline(knot_sequences, control_points)


def main():
    u = np.array([0, 0, 0, 0.3, 0.7, 1, 1, 1])
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
    cubic_b_splie = CubicBSpline.interpolate(u, d)
    cubic_b_splie.plot()


main()