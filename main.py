import argparse
import time
import numpy as np
import numpy.random
import scipy.optimize
import math
import matplotlib.pyplot as plt

from enum import Enum

# solve the maxmin game for the given payoff matrix
# this function is used to compute the equilibrium strategy
# of the row player
# solves the linear program
# max  v
# s.t. A^T x >= v
#      1^T x = 1
#      x >= 0
# @arg A The payoff matrix, such that A[i, j] is the payoff of the
# row player when he plays the strategy i and the column player plays
# the strategy j
# @returns the strategy profile and the payoff of the row player
def solve_maxmin(A : np.ndarray):
    # we need to compile the coefficient arrays correctly for
    # scipy.optimize.linprog

    # we have m + 1 vars (strategies + the maxmin value)
    n, m = A.T.shape

    # objective function, negative because linprog solves a minimization problem
    c = np.zeros(m + 1)
    c[-1] = -1.

    # constraints

    # sign is changed to get an upper bound for the inequality constraints
    A_ub = np.append(- A.T, np.ones((n, 1)), 1)
    b_ub = np.zeros(n)

    # 1 ^ T x = 1 (but we need the trailing zero for v)
    A_eq = np.append(np.ones((1, m)), np.zeros((1, 1)), 1)
    b_eq = np.ones((1))

    # strategy variables are non-negative, v is free
    bounds = [(0, None) for i in range(m)] + [(None, None)]

    result = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method='highs-ipm')

    #print(result)

    return result.x[0:m], result.x[m]


# solve the minmax game for the given payoff matrix
# this function is used to compute the equilibrium strategy
# of the column player
# solves the linear program
# min  w
# s.t. A x <= w
#      1^T x = 1
#      x >= 0
# @arg A The payoff matrix, such that A[i, j] is the payoff of the
#   row player when he plays the strategy i and the column player plays
#   the strategy j
# @returns the strategy profile of the column player and the payoff it
#   inflicts to the row player
def solve_minmax(A : np.ndarray):
    # we need to compile the coefficient arrays correctly for
    # scipy.optimize.linprog

    # we have m + 1 vars (strategies + the minmax value)
    n, m = A.shape

    # objective function, negative because linprog solves a minimization problem
    c = np.zeros(m + 1)
    c[-1] = 1.

    # upper bound constraints
    A_ub = np.append(A, -np.ones((n, 1)), 1)
    b_ub = np.zeros(n)

    # 1 ^ T x = 1 (but we need the trailing zero for w)
    A_eq = np.append(np.ones((1, m)), np.zeros((1, 1)), 1)
    b_eq = np.ones((1))

    # strategy variables are non-negative, w is free
    bounds = [(0, None) for i in range(m)] + [(None, None)]

    result = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method='highs-ipm')

    return result.x[0:m], result.x[m]


# solves a 2-player game using regret minimization against a best response
# the row player uses regret minimization, the column player computes a best response
def solve_rm_br(A_: np.ndarray, seed: int):
    # rescale the problem so that payoffs are in [0, 1] to be compatible with the Exp3 RM update rule...
    if A_.max() == A_.min():
        A = np.ones_like(A_)
    else:
        A = (A_ - A_.min()) / (A_.max() - A_.min())

    rows = A.shape[0]
    cols = A.shape[1]

    max_iter = 50

    gamma = 0.1
    #eta = math.log(1 + math.sqrt((2 * math.log(2, math.e)) / max_iter), math.e)
    eta = 0.001

    x_row = np.array([1. / rows] * rows, dtype=np.float64)

    # initialize regrets to zero (uniform distribution)
    r_row = np.array([0] * rows, dtype=np.float64)
    br_col = 0

    rng = np.random.default_rng(seed)

    # br updates
    for i in range(max_iter):
        # compute best response for each player
        col_values = A.T @ x_row
        br_col = np.argmin(col_values)

        # update strategy with regret minimization
        for i in range(4 * rows):
            # compute distributions
            exp_row = np.exp(eta * r_row)
            x_row = exp_row / np.sum(exp_row)

            # sample actions
            p_row = (1 - gamma) * x_row + np.full(rows, gamma / rows)
            #act_row = rng.choice(rows, p=p_row)

            # just to speed things up a bit, draw uniformly and only use p_row to weight the reward
            act_row = rng.integers(rows)

            # observe reward (against best response computed at the beginning)
            reward_row = A[act_row, br_col]

            # update cumulative rewards
            r_row[act_row] += (reward_row / p_row[act_row])
            assert math.isfinite(sum(np.exp(eta * r_row)))


    x_col = np.array([1.0 if i == br_col else 0.0 for i in range(cols)])
    return x_row, x_col, x_row.T @ A_ @ x_col


# solve a 2-player zero-sum game with the double-oracle algorithm
# tabular form, matrix A is the payoff of the row player
def double_oracle(A: np.ndarray):
    rows = A.shape[0]
    cols = A.shape[1]

    # initialize arrays of row/column flags (if true, then the corresponding strategy is in the population)
    row_flags = [True] + (rows - 1) * [False]
    col_flags = [True] + (cols - 1) * [False]

    # initialize lists of available strategies
    row_strategies = [i for i in range(rows) if row_flags[i]]
    col_strategies = [i for i in range(cols) if col_flags[i]]

    n = 0

    iv = []
    ivr = []
    ivc = []

    while True:
        n = n + 1

        # solve restricted game
        Ar = A[np.ix_(row_strategies, col_strategies)]
        xr_row, v_row = solve_maxmin(Ar)
        xr_col, v_col = solve_minmax(Ar)

        # extend restricted row strategy
        assert len(xr_row) == len(row_strategies)
        x_row = np.zeros(A.shape[0])
        for i in range(len(xr_row)):
            x_row[row_strategies[i]] = xr_row[i]

        # extend restricted col strategy
        assert len(xr_col) == len(col_strategies)
        x_col = np.zeros(A.shape[1])
        for i in range(len(xr_col)):
            x_col[col_strategies[i]] = xr_col[i]

        # compute response values for the restricted strategies
        row_values = A @ x_col
        col_values = A.T @ x_row

        iv.append(x_row.T @ (A @ x_col))
        ivr.append(row_values.max())
        ivc.append(col_values.min())

        updated = False

        # add best responses

        # max val for the row player
        vr = row_values.max()
        for i in range(len(row_values)):
            if np.isclose(row_values[i], vr) and row_flags[i] is False:
                row_strategies.append(i)
                row_flags[i] = True
                updated = True
                break

        # min val for the column player
        vc = col_values.min()
        for i in range(len(col_values)):
            if np.isclose(col_values[i], vc) and col_flags[i] is False:
                col_strategies.append(i)
                col_flags[i] = True
                updated = True
                break

        if not updated:
            return x_row, x_col, vr, vc, n, iv, ivr, ivc


class PopulationGrowthStrategy(Enum):
    RangeOfSkill = 1
    AnytimeDoubleOracle = 2


def double_oracle_with_partially_unrestricted_subgames(A: np.ndarray, population_growth : PopulationGrowthStrategy):
    rows = A.shape[0]
    cols = A.shape[1]

    # initialize arrays of row/column flags (if true, then the corresponding strategy is in the population)
    row_flags = [True] + (rows - 1) * [False]
    col_flags = [True] + (cols - 1) * [False]

    # initialize lists of available strategies
    row_strategies = [i for i in range(rows) if row_flags[i]]
    col_strategies = [i for i in range(cols) if col_flags[i]]

    n = 0

    iv = []
    ivr = []
    ivc = []

    while True:
        n = n + 1

        # compute generalized best response (GBR) of the row player (column player is restricted)
        Ar_row = A[np.ix_(range(rows), col_strategies)]
        # solve iteratively adding rows (which are unrestricted)
        #gbr_row_r, gbr_row_c, gbr_row_vr, gbr_row_vc, _ = double_oracle(Ar_row, restrict_cols=False)
        gbr_row_r, gbr_row_vr = solve_maxmin(Ar_row)
        gbr_row_c, gbr_row_vc = solve_minmax(Ar_row)

        # compute generalized best response of the column player (row player is restricted)
        Ar_col = A[np.ix_(row_strategies, range(cols))]
        #gbr_col_r, gbr_col_c, gbr_col_vr, gbr_col_vc, _ = double_oracle(Ar_col, restrict_rows=False)
        gbr_col_r, gbr_col_vr = solve_maxmin(Ar_col)
        gbr_col_c, gbr_col_vc = solve_minmax(Ar_col)

        updated = False

        # extend restricted strategies
        # assert len(gbr_col_r) == len(row_strategies)
        x_row = np.zeros(rows)
        for i in range(len(gbr_col_r)):
            x_row[row_strategies[i]] = gbr_col_r[i]

        # assert len(gbr_row_c) == len(col_strategies)
        x_col = np.zeros(cols)
        for i in range(len(gbr_row_c)):
            x_col[col_strategies[i]] = gbr_row_c[i]

        # compute strategy values
        row_values = A @ x_col
        col_values = A.T @ x_row

        iv.append(x_row.T @ (A @ x_col))
        ivr.append(row_values.max())
        ivc.append(col_values.min())

        # if policy is ROS, add the generalized strategy support to the population
        if population_growth == PopulationGrowthStrategy.RangeOfSkill:
            # add the support of the row player's GBR
            for i in range(rows):
                if gbr_row_r[i] > 0 and row_flags[i] is False:
                    row_flags[i] = True
                    row_strategies.append(i)
                    updated = True

            # and similarly for the column player
            for i in range(cols):
                if gbr_col_c[i] > 0 and col_flags[i] is False:
                    col_flags[i] = True
                    col_strategies.append(i)
                    updated = True

            #if not updated:
            #    return gbr_row_r, gbr_col_c, gbr_row_vr, gbr_col_vc
        # if policy is ADO, add a best response to the restricted opponent for each player
        elif population_growth == PopulationGrowthStrategy.AnytimeDoubleOracle:
            # add best responses to the populations
            vr = row_values.max()
            for i in range(rows):
                if np.isclose(row_values[i], vr) and row_flags[i] is False:
                    row_strategies.append(i)
                    row_flags[i] = True
                    updated = True
                    break

            vc = col_values.min()
            for i in range(cols):
                if np.isclose(col_values[i], vc) and col_flags[i] is False:
                    col_strategies.append(i)
                    col_flags[i] = True
                    updated = True
                    break
        else:
            raise ValueError(f'Invalid population growth strategy enum \'{population_growth}\', accepted values are '
                             f'{PopulationGrowthStrategy.RangeOfSkill}, '
                             f'\'{PopulationGrowthStrategy.AnytimeDoubleOracle}\'')

        if not updated:
            return gbr_row_r, gbr_col_c, gbr_row_vr, gbr_col_vc, n, iv, ivr, ivc


# Like ADO/ROS, but the restricted player uses regret minimization against a best response
# this ensures that the restricted strategy is least-exploitable (up to the approximation error
# of the regret-minimizing algorithm) without solving the restricted game at the equilibrium, which
# may be intractable for large games
def rm_br_do(A: np.ndarray, min_iterations = None):
    rows = A.shape[0]
    cols = A.shape[1]

    # initialize arrays of row/column flags (if true, then the corresponding strategy is in the population)
    row_flags = [True] + (rows - 1) * [False]
    col_flags = [True] + (cols - 1) * [False]

    # initialize lists of available strategies
    row_strategies = [i for i in range(rows) if row_flags[i]]
    col_strategies = [i for i in range(cols) if col_flags[i]]

    n = 0

    iv = []
    ivr = []
    ivc = []

    seed_generator = np.random.default_rng(12345)

    while True:
        n = n + 1

        # solve the column-restricted game with rm_br
        Ar_row = A[np.ix_(range(rows), col_strategies)]
        rmbr_row_c, rmbr_row_r, rmbr_row_v = solve_rm_br(-Ar_row.T, seed_generator.integers(low=0, high=50000))

        # solve the row-restricted game with rm_br
        Ar_col = A[np.ix_(row_strategies, range(cols))]
        rmbr_col_r, rmbr_col_c, rmbr_col_v = solve_rm_br(Ar_col, seed_generator.integers(low=0, high=50000))

        updated = False

        # extend restricted strategies
        x_row = np.zeros(rows) # restricted NE row strategy
        for i in range(len(rmbr_col_r)):
            x_row[row_strategies[i]] = rmbr_col_r[i]

        #assert len(gbr_row_c) == len(col_strategies)
        x_col = np.zeros(cols) # restricted NE column strategy
        for i in range(len(rmbr_row_c)):
            x_col[col_strategies[i]] = rmbr_row_c[i]

        # compute strategy values
        row_values = A @ x_col
        col_values = A.T @ x_row

        iv.append(x_row.T @ (A @ x_col))
        ivr.append(row_values.max())
        ivc.append(col_values.min())

        # add best responses to the populations
        vr = row_values.max()
        for i in range(rows):
            if np.isclose(row_values[i], vr, rtol=1e-2) and row_flags[i] is False:
                row_strategies.append(i)
                row_flags[i] = True
                updated = True
                break

        vc = col_values.min()
        for i in range(cols):
            if np.isclose(col_values[i], vc, rtol=1e-2) and col_flags[i] is False:
                col_strategies.append(i)
                col_flags[i] = True
                updated = True
                break

        if not updated and (min_iterations is None or n >= min_iterations):
            print(f'RM-BR-DO row population size at convergence: {len(row_strategies)}, row value: {vr}')
            print(f'RM-BR-DO column population size at convergence: {len(col_strategies)}, col value: {vc}')
            return x_row, x_col, vr, vc, n, iv, ivr, ivc


def test_solvers(A : np.ndarray, tolerance: float = 1e-10):
    print('Computing exact game value with LP... ', end='')
    x_row_gt, v_row_gt = solve_maxmin(A)
    x_col_gt, v_col_gt = solve_minmax(A)

    print(f'Game value is {v_row_gt}')

    # solve with double oracle solver
    print('Solving with Double Oracle...')
    t = time.perf_counter()
    x_row_do, x_col_do, v_row_do, v_col_do, n_do, iv_do, ivr_do, ivc_do = double_oracle(A)
    t1 = time.perf_counter()
    print(f'Solving DO took {t1 - t:.3f} seconds')
    print(f'Row value = {v_row_do}, column value = {v_col_do}, gap = {v_row_do - v_col_do}')

    ## solve with range-of-skill solver
    print('Solving with Range-Of-Skill')
    t = time.perf_counter()
    x_row_ros, x_col_ros, v_row_ros, v_col_ros, n_ros, iv_ros, ivr_ros, ivc_ros =\
        double_oracle_with_partially_unrestricted_subgames(A, PopulationGrowthStrategy.RangeOfSkill)
    t1 = time.perf_counter()
    print(f'Gap between game values is {v_row_do - v_col_do}')
    print(f'Row value = {v_row_ros}, column value = {v_col_ros}, gap = {v_row_ros - v_col_ros}')

    ## solve with anytime double oracle solver
    print('Solving with Anytime Double Oracle')
    t = time.perf_counter()
    x_row_ado, x_col_ado, v_row_ado, v_col_ado, n_ado, iv_ado, ivr_ado, ivc_ado = \
        double_oracle_with_partially_unrestricted_subgames(A, PopulationGrowthStrategy.AnytimeDoubleOracle)
    t1 = time.perf_counter()
    print(f'Row value = {v_row_ado}, column value = {v_col_ado}, gap = {v_row_ado - v_col_ado}')
    print(f'Solving ADO took {t1 - t:.3f} seconds')

    print('Solving with Regret-Minimization-against-a-Best-Response Double Oracle (not exact)')
    t = time.perf_counter()
    ## solve with rm-br-do solver
    x_row_rmdo, x_col_rmdo, v_row_rmdo, v_col_rmdo, n_rmdo, iv_rmdo, ivr_rmdo, ivc_rmdo = rm_br_do(A, min_iterations=n_ado)
    t1 = time.perf_counter()
    print(f'Row value = {v_row_rmdo}, column value = {v_col_rmdo}, gap = {v_row_rmdo - v_col_rmdo}')
    print(f'Solving RM-BR-DO took {t1 - t:.3f} seconds')

    # note: the exploitability is computed in this way because the column values use the
    # same payoff matrix as the row value, hence the signs in the (2-player) summation must
    # be inverted for the column player, resulting in being simply the difference between the
    # row and colum player best responses at each iteration
    v = v_row_gt
    e_do = np.array(ivr_do) - np.array(ivc_do)
    e_ros = np.array(ivr_ros) - np.array(ivc_ros)
    e_ado = np.array(ivr_ado) - np.array(ivc_ado)
    e_rmdo = np.array(ivr_rmdo) - np.array(ivc_rmdo)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(e_do, label='DoubleOracle')
    ax.plot(e_ros, label='RangeOfSkill')
    ax.plot(e_ado, label='AnytimeDoubleOracle')
    ax.plot(e_rmdo, label='RM-BR DoubleOracle')

    ax.set_ylabel('Exploitability')
    ax.set_xlabel('Iterations')
    ax.legend()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axhline(v, ls='-', color='black', lw=1, alpha=0.7)
    ax.text(0.8 * n_ado, v + 0.02, f'Equilibrium = {v:.3f}')
    ax.plot(iv_do, label='DoubleOracle')
    ax.plot(iv_ros, label='RangeOfSkill')
    ax.plot(iv_ado, label='AnytimeDoubleOracle')
    ax.plot(iv_rmdo, label='RM-BR DoubleOracle')

    ax.set_ylabel('Game value')
    ax.set_xlabel('Iterations')
    ax.legend()

    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rows', type=int, default=100, help='The rows of the random game matrix')
parser.add_argument('-c', '--cols', type=int, default=100, help='The columns of the random game matrix')
parser.add_argument('-s', '--seed', type=int, default=1, help='The RNG seed')

args = parser.parse_args()

np.set_printoptions(precision=5)

rng = np.random.default_rng(args.seed)
test_solvers(rng.uniform(0, 1, (args.rows, args.cols)), tolerance=1e-7)
