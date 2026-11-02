# Eigenvectors and eigenvalues (Math for ML, week 4) — SAMPLE NOTES

## Geometric meaning
A matrix A is a linear transformation: it moves every vector somewhere else.
Most vectors get knocked off their original line. An eigenvector v is a
direction that A does not rotate: A only stretches or shrinks it, so
A v = λ v. The eigenvalue λ is the stretch factor. If λ is negative the vector
flips to point the other way but stays on the same line.

Picture a grid being sheared or stretched. The eigenvectors are the lines
through the origin that map onto themselves.

## Finding eigenvalues of a 2x2 matrix by hand
1. Write A - λI.
2. Set the determinant to zero: det(A - λI) = 0. This is the characteristic
   polynomial.
3. For A = [[a, b], [c, d]] this is λ² - (a + d)λ + (ad - bc) = 0.
   The trace is a + d and the determinant is ad - bc.
4. Solve the quadratic for λ. Each root is an eigenvalue.
5. For each λ, solve (A - λI) v = 0 to get the eigenvector.

Example: A = [[2, 1], [1, 2]]. λ² - 4λ + 3 = 0, so λ = 1 and λ = 3.
For λ = 3, v = [1, 1]. For λ = 1, v = [1, -1].

## Why it matters for ML
PCA finds the eigenvectors of the covariance matrix. The eigenvector with the
largest eigenvalue is the direction of most variance in the data.
