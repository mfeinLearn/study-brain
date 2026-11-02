# Determinants and invertibility — SAMPLE NOTES

The determinant of a matrix tells you how much the linear transformation
scales area (2D) or volume (3D). det = 2 means areas double. A negative
determinant means orientation flips (like a mirror).

A matrix is not invertible (singular) when its determinant is 0. That means
the transformation squashes space into a lower dimension, for example a plane
onto a line, so you cannot undo it. Equivalent statements:
- det(A) = 0
- the columns are linearly dependent
- rank is less than n
- 0 is an eigenvalue of A
