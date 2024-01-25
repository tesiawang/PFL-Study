import cvxpy as cp
import numpy as np

n = 5
p = np.ones(n)
G = - np.identity(n)
h = np.zeros(n)
A = np.ones((1, n))
b = np.ones(1)
d = np.ones(n)
lamba = 0.8



# Bug: some convex operations must be done with cvxpy built-in functions, e.g., cp.norm
# "power(Sum(power(var1, 2.0), None, False), 0.5)" is not DCP
# "cp.norm(var1,2)" is DCP


x = cp.Variable(5)
x_prime = cp.multiply(x,1/p)  # element-wise multiplication
prob = cp.Problem(cp.Minimize(cp.norm(x_prime,2) + d.T @ x),
                  [G @ x <= h,
                   A @ x == b]
                  )

prob.solve(solver = cp.ECOS)



# P = lamba * np.identity(n)
# P = cp.atoms.affine.wraps.psd_wrap(P)
# q = d - 2 * lamba * p
#
# prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
#                   [G @ x <= h,
#                    A @ x == b]
#                   )

prob.solve()


print(x.value)