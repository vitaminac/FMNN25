from abc import ABC, abstractmethod
from functools import cached_property, wraps
import inspect
import time
from typing import Union

import numpy as np
import numpy.typing
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

from mpi4py import MPI

"""
Get a communicator :
The most common communicator is the
one that connects all available processes
which is called COMM_WORLD .
Clone the communicator to avoid interference
with other libraries or applications
"""
comm = MPI.Comm.Clone(MPI.COMM_WORLD)
RANK_OF_CURRENT_NODE = comm.Get_rank()
NUMBER_OF_NODES = comm.Get_size()


def timeit(fn):
    @wraps(fn)
    def timeit_wrapper(*args, **kwargs):
        is_method = False
        try:
            is_method = inspect.getfullargspec(fn)[0][0] == 'self'
        except:
            pass
        start_time = time.perf_counter()
        result = fn(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        if is_method:
            print(
                f'Method {args[0].__class__.__name__}#{fn.__name__}{args} {kwargs} took {total_time:.4f} seconds on the node {RANK_OF_CURRENT_NODE}')
        else:
            print(
                f'Function {fn.__name__}{args} {kwargs} took {total_time:.4f} seconds on the node {RANK_OF_CURRENT_NODE}')
        return result
    return timeit_wrapper


class LaplaceEquation2D(ABC):
    def __init__(self, width: Union[int, float], height: Union[int, float], h: float) -> None:
        self.width = width
        self.height = height
        self._h = h

    @abstractmethod
    def solve(self) -> np.typing.NDArray[np.float_]:
        raise NotImplementedError()

    def set_neuman_condition(self,  i: int, j: int, y_prime: float) -> None:
        assert i == self.topmost_i or i == self.bottommost_i or j == self.leftmost_j or j == self.rightmost_j, "(i, j) must be at the boundary"
        index = self._index(i, j)
        self._set_A_row_to_zeros(index)
        n_neighbors = 0
        if i < self.topmost_i:
            self._A[index, self._index(i + 1, j)] = 1.0 / self._h**2
            n_neighbors += 1
        if i > self.bottommost_i:
            self._A[index, self._index(i - 1, j)] = 1.0 / self._h**2
            n_neighbors += 1
        if j > self.leftmost_j:
            self._A[index, self._index(i, j - 1)] = 1.0 / self._h**2
            n_neighbors += 1
        if j < self.rightmost_j:
            self._A[index, self._index(i, j + 1)] = 1.0 / self._h**2
            n_neighbors += 1
        self._A[index, index] = -n_neighbors / self._h**2
        self._b[index] = -y_prime / self._h

    def set_dirichlet_condition(self,  i: int, j: int, y: float) -> None:
        assert i == self.topmost_i or i == self.bottommost_i or j == self.leftmost_j or j == self.rightmost_j, "(i, j) must be at the boundary"
        index = self._index(i, j)
        self._set_A_row_to_zeros(index)
        self._A[index, index] = 1
        self._b[index] = y

    def _set_A_row_to_zeros(self, row_index: int):
        self._set_matrix_row_to_zeros(self._A, row_index)

    @abstractmethod
    def _set_matrix_row_to_zeros(self, A, row_index: int):
        raise NotImplementedError()

    @cached_property
    def _b(self):
        return np.zeros(self.size, dtype=np.double)

    def _index(self, i: int, j: int) -> int:
        return i * self.nx + j

    def _set_boundary_equation_to_empty(self, A) -> None:
        for i in self.iterate_index_vertically_from_bottom_to_top():
            self._set_matrix_row_to_zeros(A, self._index(i, self.leftmost_j))
            self._set_matrix_row_to_zeros(A, self._index(i, self.rightmost_j))
        for j in self.iterate_index_horizontally_from_left_to_right():
            self._set_matrix_row_to_zeros(A, self._index(self.topmost_i, j))
            self._set_matrix_row_to_zeros(A, self._index(self.bottommost_i, j))

    def iterate_index_horizontally_from_left_to_right(self):
        return self.iterate(self.leftmost_j, self.rightmost_j)

    def iterate_index_horizontally_from_left_to_middle(self):
        return self.iterate(self.leftmost_j, self.middle_j)

    def iterate_index_vertically_from_bottom_to_middle(self):
        return self.iterate(self.bottommost_i, self.middle_i)

    def iterate_index_vertically_from_middle_to_top(self):
        return self.iterate(self.middle_i, self.topmost_i)

    def iterate_index_vertically_from_bottom_to_top(self):
        return self.iterate(self.bottommost_i, self.topmost_i)

    def iterate(self, first_index, last_index):
        if first_index <= last_index:
            return range(first_index, last_index + 1)
        else:
            return range(first_index, last_index - 1, -1)

    @cached_property
    def size(self) -> int:
        return self.ny * self.nx

    @cached_property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @cached_property
    def nx(self) -> int:
        return int(self.width/self._h + 1)

    @cached_property
    def ny(self) -> int:
        return int(self.height/self._h + 1)

    @property
    def topmost_i(self) -> int:
        return self.ny - 1

    @property
    def bottommost_i(self) -> int:
        return 0

    @property
    def middle_i(self) -> int:
        return (self.topmost_i + self.bottommost_i) // 2

    @property
    def leftmost_j(self) -> int:
        return 0

    @property
    def rightmost_j(self) -> int:
        return self.nx - 1

    @property
    def middle_j(self) -> int:
        return (self.leftmost_j + self.rightmost_j) // 2


class LaplaceEquationDense2D(LaplaceEquation2D):
    def __init__(self, width: Union[int, float], height: Union[int, float], h: float) -> None:
        super().__init__(width, height, h)

    def solve(self) -> np.typing.NDArray[np.float_]:
        return scipy.linalg.solve(self._A, self._b).reshape(self.shape)

    def _set_matrix_row_to_zeros(self, A, row_index: int):
        A[row_index, :] = 0

    @cached_property
    def _A(self):
        c = np.zeros(self.size, dtype=np.double)
        c[self._index(0, 0)] = -4.0
        c[self._index(0, 1)] = c[self._index(1, 0)] = 1.0
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html
        A = scipy.linalg.toeplitz(c) / self._h**2
        self._set_boundary_equation_to_empty(A)
        return A


class LaplaceEquationSparse2D(LaplaceEquation2D):
    def __init__(self, width: Union[int, float], height: Union[int, float], h: float) -> None:
        super().__init__(width, height, h)

    def solve(self) -> np.typing.NDArray[np.float_]:
        return scipy.sparse.linalg.spsolve(self._A, self._b).reshape(self.shape)

    def _set_matrix_row_to_zeros(self, A, row_index: int):
        # same result as self._A[index, :] = 0 but it is efficient for csr matrix
        from_index = A.indptr[row_index]
        end_index = A.indptr[row_index + 1]
        A.data[from_index:end_index] = 0

    @cached_property
    def _A(self):
        i_plus = self._index(1, 0)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
        diagonals = np.array([1.0, 1.0, -4.0, 1.0, 1.0]) / self._h**2
        A = scipy.sparse.diags(diagonals=diagonals,
                               offsets=(-i_plus, -1, 0, 1, i_plus),
                               shape=(self.size, self.size),
                               format='csr',
                               dtype=np.double)
        self._set_boundary_equation_to_empty(A)
        return A


def plot(U, width, height, nx, ny, plot_name):
    x = np.linspace(0, width, nx)
    y = np.linspace(0, height, ny)
    X, Y = np.meshgrid(x, y)
    plt.figure()
    cp = plt.contourf(X, Y, U, levels=1000)
    plt.axis('scaled')
    plt.title(plot_name)
    plt.colorbar(cp)
    plt.show()


TEMPERATURE_OF_NORMAL_WALL = 15
TEMPERATURE_OF_WALL_WITH_HEATER = 40
TEMPERATURE_OF_WALL_WITH_BIG_WINDOW = 5
LAPLACE_2D_DISCRETIZATION_DELTA = 1.0 / 20
RANK_FOR_MAIN_CONTROL = 0 % NUMBER_OF_NODES
RANK_FOR_MIDDLE_BIG_ROOM = 1 % NUMBER_OF_NODES
RANK_FOR_LEFT_LITTLE_ROOM = 2 % NUMBER_OF_NODES
RANK_FOR_RIGHT_LITTLE_ROOM = 3 % NUMBER_OF_NODES


@timeit
def create_middle_big_room(LaplaceEquation2DClass):
    middle_big_room = LaplaceEquation2DClass(
        1, 2, LAPLACE_2D_DISCRETIZATION_DELTA)
    if RANK_OF_CURRENT_NODE == RANK_FOR_MIDDLE_BIG_ROOM:
        # set boundary condition for middle big room
        for j in middle_big_room.iterate_index_horizontally_from_left_to_right():
            # top wall with heater
            middle_big_room.set_dirichlet_condition(
                middle_big_room.topmost_i, j, TEMPERATURE_OF_WALL_WITH_HEATER)
            # bottom wall with big window
            middle_big_room.set_dirichlet_condition(
                middle_big_room.bottommost_i, j, TEMPERATURE_OF_WALL_WITH_BIG_WINDOW)
        for i in middle_big_room.iterate_index_vertically_from_middle_to_top():
            # left top half normal wall
            middle_big_room.set_dirichlet_condition(
                i, middle_big_room.leftmost_j, TEMPERATURE_OF_NORMAL_WALL)
        for i in middle_big_room.iterate_index_vertically_from_bottom_to_middle():
            # right bottom half normal wall
            middle_big_room.set_dirichlet_condition(
                i, middle_big_room.rightmost_j, TEMPERATURE_OF_NORMAL_WALL)
    return middle_big_room


@timeit
def create_left_little_room(LaplaceEquation2DClass):
    left_little_room = LaplaceEquation2DClass(
        1, 1, LAPLACE_2D_DISCRETIZATION_DELTA)
    if RANK_OF_CURRENT_NODE == RANK_FOR_LEFT_LITTLE_ROOM:
        # set boundary condition for left little room
        for j in left_little_room.iterate_index_horizontally_from_left_to_right():
            # top normal wall
            left_little_room.set_dirichlet_condition(
                left_little_room.topmost_i, j, TEMPERATURE_OF_NORMAL_WALL)
            # bottom normal wall
            left_little_room.set_dirichlet_condition(
                left_little_room.bottommost_i, j, TEMPERATURE_OF_NORMAL_WALL)
        for i in left_little_room.iterate_index_vertically_from_bottom_to_top():
            # left wall with heater
            left_little_room.set_dirichlet_condition(
                i, left_little_room.leftmost_j, TEMPERATURE_OF_WALL_WITH_HEATER)
    return left_little_room


@timeit
def create_right_little_room(LaplaceEquation2DClass):
    right_little_room = LaplaceEquation2DClass(
        1, 1, LAPLACE_2D_DISCRETIZATION_DELTA)
    if RANK_OF_CURRENT_NODE == RANK_FOR_RIGHT_LITTLE_ROOM:
        for i in right_little_room.iterate_index_vertically_from_bottom_to_top():
            # right wall with heater
            right_little_room.set_dirichlet_condition(
                i, right_little_room.rightmost_j, TEMPERATURE_OF_WALL_WITH_HEATER)
        # set boundary condition for right little room"
        for j in right_little_room.iterate_index_horizontally_from_left_to_right():
            # top normal wall
            right_little_room.set_dirichlet_condition(
                right_little_room.topmost_i, j, TEMPERATURE_OF_NORMAL_WALL)
            # bottom normal wall
            right_little_room.set_dirichlet_condition(
                right_little_room.bottommost_i, j, TEMPERATURE_OF_NORMAL_WALL)
    return right_little_room


TAG_FOR_GAMMA_1_PRIME_ON_STEP_1 = 87
TAG_FOR_GAMMA_2_PRIME_ON_STEP_1 = 88
TAG_FOR_NEW_GAMMA_1_ON_STEP_2 = 97
TAG_FOR_NEW_GAMMA_2_ON_STEP_2 = 98
TAG_FOR_FINAL_TEMPERATURE_OF_MIDDLE_BIG_ROOM = 100
TAG_FOR_FINAL_TEMPERATURE_OF_LEFT_LITTLE_ROOM = 101
TAG_FOR_FINAL_TEMPERATURE_OF_RIGHT_LITTLE_ROOM = 102


def obtain_initial_guess_for_Gamma1_and_Gamma2_for_Omega2(middle_big_room) -> tuple[np.typing.NDArray[np.float_], np.typing.NDArray[np.float_]]:
    Gamma_1 = None
    Gamma_2 = None
    if RANK_OF_CURRENT_NODE == RANK_FOR_MIDDLE_BIG_ROOM:
        Gamma_1 = np.zeros(len(
            middle_big_room.iterate_index_vertically_from_bottom_to_middle()), dtype=np.double)
        Gamma_2 = np.zeros(len(
            middle_big_room.iterate_index_vertically_from_middle_to_top()), dtype=np.double)
    return Gamma_1, Gamma_2


def solve_Omega2_with_Gamma1_and_Gamma2_and_get_Gamma1_prime_and_Gamma2_prime(middle_big_room, Gamma_1, Gamma_2) -> tuple[np.typing.NDArray[np.float_], np.typing.NDArray[np.float_]]:
    middle_big_room_temperature_U = None
    Gamma_1_prime = None
    Gamma_2_prime = None
    if RANK_OF_CURRENT_NODE == RANK_FOR_MIDDLE_BIG_ROOM:
        # update Dirichlet conditions for \Omega_2 at \Gamma_1 and \Gamma_2
        for middle_big_room_i, u_DC in zip(middle_big_room.iterate_index_vertically_from_bottom_to_middle(), Gamma_1):
            middle_big_room.set_dirichlet_condition(
                middle_big_room_i, middle_big_room.leftmost_j, u_DC)
        for middle_big_room_i, u_DC in zip(middle_big_room.iterate_index_vertically_from_middle_to_top(), Gamma_2):
            middle_big_room.set_dirichlet_condition(
                middle_big_room_i, middle_big_room.rightmost_j, u_DC)
        # solve \Omage_2
        middle_big_room_temperature_U = middle_big_room.solve()
        # get \Gamma_1' and \Gamma_2'
        Gamma_1_prime = middle_big_room_temperature_U[middle_big_room.iterate_index_vertically_from_bottom_to_middle(),
                                                      middle_big_room.leftmost_j + 1] - Gamma_1
        Gamma_2_prime = Gamma_2 - middle_big_room_temperature_U[middle_big_room.iterate_index_vertically_from_middle_to_top(),
                                                                middle_big_room.rightmost_j - 1]
        if RANK_OF_CURRENT_NODE != RANK_FOR_LEFT_LITTLE_ROOM:
            comm.Send([Gamma_1_prime, Gamma_1_prime.size, MPI.DOUBLE],
                      dest=RANK_FOR_LEFT_LITTLE_ROOM, tag=TAG_FOR_GAMMA_1_PRIME_ON_STEP_1)
        if RANK_OF_CURRENT_NODE != RANK_FOR_RIGHT_LITTLE_ROOM:
            comm.Send([Gamma_2_prime, Gamma_2_prime.size, MPI.DOUBLE],
                      dest=RANK_FOR_RIGHT_LITTLE_ROOM, tag=TAG_FOR_GAMMA_2_PRIME_ON_STEP_1)
    else:
        if RANK_OF_CURRENT_NODE == RANK_FOR_LEFT_LITTLE_ROOM:
            Gamma_1_prime = np.empty(len(
                middle_big_room.iterate_index_vertically_from_bottom_to_middle()), dtype=np.double)
            comm.Recv([Gamma_1_prime, Gamma_1_prime.size, MPI.DOUBLE],
                      source=RANK_FOR_MIDDLE_BIG_ROOM, tag=TAG_FOR_GAMMA_1_PRIME_ON_STEP_1)
        if RANK_OF_CURRENT_NODE == RANK_FOR_RIGHT_LITTLE_ROOM:
            Gamma_2_prime = np.empty(len(
                middle_big_room.iterate_index_vertically_from_middle_to_top()), dtype=np.double)
            comm.Recv([Gamma_2_prime, Gamma_2_prime.size, MPI.DOUBLE],
                      source=RANK_FOR_MIDDLE_BIG_ROOM, tag=TAG_FOR_GAMMA_2_PRIME_ON_STEP_1)
    return middle_big_room_temperature_U, Gamma_1_prime, Gamma_2_prime


def solve_Omega1_with_Gamma1_prime_and_get_new_Gamma1(left_little_room, Gamma_1_prime) -> np.typing.NDArray[np.float_]:
    left_little_room_temperature_U = None
    new_Gamma_1 = None
    if RANK_OF_CURRENT_NODE == RANK_FOR_LEFT_LITTLE_ROOM:
        # update \Omega_1 with Neumann conditions at \Gamma_1
        for left_little_room_i, u_NC in zip(
                left_little_room.iterate_index_vertically_from_bottom_to_top(),
                Gamma_1_prime):
            left_little_room.set_neuman_condition(
                left_little_room_i, left_little_room.rightmost_j, u_NC)
        # solve \Omage_1
        left_little_room_temperature_U = left_little_room.solve()
        # get new \Gamma_1
        new_Gamma_1 = left_little_room_temperature_U[left_little_room.iterate_index_vertically_from_bottom_to_top(),
                                                     left_little_room.rightmost_j]
        if RANK_OF_CURRENT_NODE != RANK_FOR_MIDDLE_BIG_ROOM:
            comm.Send([new_Gamma_1, new_Gamma_1.size, MPI.DOUBLE],
                      dest=RANK_FOR_MIDDLE_BIG_ROOM, tag=TAG_FOR_NEW_GAMMA_1_ON_STEP_2)
    else:
        if RANK_OF_CURRENT_NODE == RANK_FOR_MIDDLE_BIG_ROOM:
            new_Gamma_1 = np.empty(len(
                left_little_room.iterate_index_vertically_from_bottom_to_top()), dtype=np.double)
            comm.Recv([new_Gamma_1, new_Gamma_1.size, MPI.DOUBLE],
                      source=RANK_FOR_LEFT_LITTLE_ROOM, tag=TAG_FOR_NEW_GAMMA_1_ON_STEP_2)
    return left_little_room_temperature_U, new_Gamma_1


def solve_Omega3_with_Gamma2_prime_and_get_new_Gamma2(right_little_room, Gamma_2_prime) -> np.typing.NDArray[np.float_]:
    right_little_room_temperature_U = None
    new_Gamma_2 = None
    if RANK_OF_CURRENT_NODE == RANK_FOR_RIGHT_LITTLE_ROOM:
        # update \Omega_3 with Neumann conditions at \Gamma_2
        for right_little_room_i, u_NC in zip(
                right_little_room.iterate_index_vertically_from_bottom_to_top(),
                Gamma_2_prime):
            right_little_room.set_neuman_condition(
                right_little_room_i, right_little_room.leftmost_j, u_NC)
        # solve \Omage_3
        right_little_room_temperature_U = right_little_room.solve()
        # get new \Gamma_2
        new_Gamma_2 = right_little_room_temperature_U[right_little_room.iterate_index_vertically_from_bottom_to_top(),
                                                      right_little_room.leftmost_j]
        if RANK_OF_CURRENT_NODE != RANK_FOR_MIDDLE_BIG_ROOM:
            comm.Send([new_Gamma_2, new_Gamma_2.size, MPI.DOUBLE],
                      dest=RANK_FOR_MIDDLE_BIG_ROOM, tag=TAG_FOR_NEW_GAMMA_2_ON_STEP_2)
    else:
        if RANK_OF_CURRENT_NODE == RANK_FOR_MIDDLE_BIG_ROOM:
            new_Gamma_2 = np.empty(len(
                right_little_room.iterate_index_vertically_from_bottom_to_top()), dtype=np.double)
            comm.Recv([new_Gamma_2, new_Gamma_2.size, MPI.DOUBLE],
                      source=RANK_FOR_RIGHT_LITTLE_ROOM, tag=TAG_FOR_NEW_GAMMA_2_ON_STEP_2)
    return right_little_room_temperature_U, new_Gamma_2


@timeit
def calculate_temperature_distribution(middle_big_room, left_little_room, right_little_room, omega=0.8):
    # calculate overall room temperature using Dirichlet-Neumann method
    Gamma_1, Gamma_2 = obtain_initial_guess_for_Gamma1_and_Gamma2_for_Omega2(
        middle_big_room)
    for _ in range(10):
        # step 1: Determine u_2^{k+1} by solving the problem on \Omega_2 with Dirichlet conditions at \Gamma1, \Gamma2 given by u_1^{k} and u_3^{k}
        middle_big_room_temperature_U, Gamma_1_prime, Gamma_2_prime = solve_Omega2_with_Gamma1_and_Gamma2_and_get_Gamma1_prime_and_Gamma2_prime(
            middle_big_room, Gamma_1, Gamma_2)
        # step 2: Determine u_1^{k+1} and u_3^{k+1} by solving the problem on \Omega_1 abd \Omega_3 with Neumann conditions at \Gamma_1 and \Gamma_2 given by u_2^{k+1}
        left_little_room_temperature_U, new_Gamma_1 = solve_Omega1_with_Gamma1_prime_and_get_new_Gamma1(
            left_little_room, Gamma_1_prime)
        right_little_room_temperature_U, new_Gamma_2 = solve_Omega3_with_Gamma2_prime_and_get_new_Gamma2(
            right_little_room, Gamma_2_prime)
        # step 3: Use relaxation: \Gamma_{k+1} <- \omega * \Gamma^{k+1} + (1 − \omega) * \Gamma^k
        if RANK_OF_CURRENT_NODE == RANK_FOR_MIDDLE_BIG_ROOM:
            Gamma_1 = omega * new_Gamma_1 + (1 - omega) * Gamma_1
            Gamma_2 = omega * new_Gamma_2 + (1 - omega) * Gamma_2

    if RANK_OF_CURRENT_NODE == RANK_FOR_MIDDLE_BIG_ROOM and RANK_OF_CURRENT_NODE != RANK_FOR_MAIN_CONTROL:
        comm.Send([middle_big_room_temperature_U, middle_big_room.size, MPI.DOUBLE],
                  dest=RANK_FOR_MAIN_CONTROL, tag=TAG_FOR_FINAL_TEMPERATURE_OF_MIDDLE_BIG_ROOM)

    if RANK_OF_CURRENT_NODE == RANK_FOR_LEFT_LITTLE_ROOM and RANK_OF_CURRENT_NODE != RANK_FOR_MAIN_CONTROL:
        comm.Send([left_little_room_temperature_U, left_little_room.size, MPI.DOUBLE],
                  dest=RANK_FOR_MAIN_CONTROL, tag=TAG_FOR_FINAL_TEMPERATURE_OF_LEFT_LITTLE_ROOM)

    if RANK_OF_CURRENT_NODE == RANK_FOR_RIGHT_LITTLE_ROOM and RANK_OF_CURRENT_NODE != RANK_FOR_MAIN_CONTROL:
        comm.Send([right_little_room_temperature_U, right_little_room.size, MPI.DOUBLE],
                  dest=RANK_FOR_MAIN_CONTROL, tag=TAG_FOR_FINAL_TEMPERATURE_OF_RIGHT_LITTLE_ROOM)

    if RANK_OF_CURRENT_NODE == RANK_FOR_MAIN_CONTROL:
        if RANK_OF_CURRENT_NODE != RANK_FOR_MIDDLE_BIG_ROOM:
            middle_big_room_temperature_U = np.empty(
                middle_big_room.shape, dtype=np.double)
            comm.Recv([middle_big_room_temperature_U, middle_big_room.size, MPI.DOUBLE],
                      source=RANK_FOR_MIDDLE_BIG_ROOM, tag=TAG_FOR_FINAL_TEMPERATURE_OF_MIDDLE_BIG_ROOM)
        if RANK_OF_CURRENT_NODE != RANK_FOR_LEFT_LITTLE_ROOM:
            left_little_room_temperature_U = np.empty(
                left_little_room.shape, dtype=np.double)
            comm.Recv([left_little_room_temperature_U, left_little_room.size, MPI.DOUBLE],
                      source=RANK_FOR_LEFT_LITTLE_ROOM, tag=TAG_FOR_FINAL_TEMPERATURE_OF_LEFT_LITTLE_ROOM)
        if RANK_OF_CURRENT_NODE != RANK_FOR_RIGHT_LITTLE_ROOM:
            right_little_room_temperature_U = np.empty(
                right_little_room.shape, dtype=np.double)
            comm.Recv([right_little_room_temperature_U, right_little_room.size, MPI.DOUBLE],
                      source=RANK_FOR_RIGHT_LITTLE_ROOM, tag=TAG_FOR_FINAL_TEMPERATURE_OF_RIGHT_LITTLE_ROOM)
        return middle_big_room_temperature_U, left_little_room_temperature_U, right_little_room_temperature_U


def plot_overall_room_temperature(middle_big_room, middle_big_room_temperature_U,
                                  left_little_room, left_little_room_temperature_U,
                                  right_little_room, right_little_room_temperature_U):
    # asseble to overall room temperature for visualization
    total_nx = left_little_room.nx + \
        middle_big_room.nx + right_little_room.nx - 2
    total_ny = middle_big_room.ny
    overall_U = np.empty((total_ny, total_nx)) * np.nan
    overall_U[0:left_little_room.ny,
              0:left_little_room.nx] = left_little_room_temperature_U
    overall_U[:,
              left_little_room.nx-1:left_little_room.nx+middle_big_room.nx-1] = middle_big_room_temperature_U
    overall_U[middle_big_room.middle_i:,
              left_little_room.nx + middle_big_room.nx - 2:] = right_little_room_temperature_U
    total_width = left_little_room.width + \
        middle_big_room.width + right_little_room.width
    total_height = middle_big_room.height
    plot(overall_U, total_width, total_height,
         total_nx, total_ny, "Overall Room Temperature")


def test_Laplace_equation(LaplaceEquation2DClass):
    middle_big_room = create_middle_big_room(LaplaceEquation2DClass)
    left_little_room = create_left_little_room(LaplaceEquation2DClass)
    right_little_room = create_right_little_room(LaplaceEquation2DClass)

    result = calculate_temperature_distribution(
        middle_big_room, left_little_room, right_little_room)

    if RANK_OF_CURRENT_NODE == RANK_FOR_MAIN_CONTROL:
        middle_big_room_temperature_U, left_little_room_temperature_U, right_little_room_temperature_U = result
        plot_overall_room_temperature(middle_big_room, middle_big_room_temperature_U,
                                      left_little_room, left_little_room_temperature_U,
                                      right_little_room, right_little_room_temperature_U)


def main():
    print(f"Node {RANK_OF_CURRENT_NODE} started")
    # test_Laplace_equation(LaplaceEquationDense2D)
    test_Laplace_equation(LaplaceEquationSparse2D)
    print(f"Node {RANK_OF_CURRENT_NODE} finished")


if __name__ == "__main__":
    # mpiexec /np 4 python laplace2d.py
    main()
