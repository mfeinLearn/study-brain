# Probability: conditional probability and Bayes — SAMPLE NOTES

Conditional probability: P(A | B) = P(A and B) / P(B).

Bayes' theorem flips the condition:
P(A | B) = P(B | A) P(A) / P(B)

Use Bayes when you know P(B | A) but you need P(A | B). Classic case: you know
how often a test is positive given disease (sensitivity), and you want the
chance of disease given a positive test. P(A) is the prior, P(B | A) is the
likelihood, and P(A | B) is the posterior.

Example: disease rate 1%, test sensitivity 95%, false positive rate 5%.
P(disease | positive) = 0.95 * 0.01 / (0.95 * 0.01 + 0.05 * 0.99) ≈ 0.16.

# Variance

Variance is the average squared distance from the mean. Standard deviation is
the square root of variance, so it is in the same units as the data.

Sample variance divides by n - 1 instead of n (Bessel's correction). The
sample mean is computed from the same data, so deviations from it are a bit
too small on average. Dividing by n - 1 makes the estimate unbiased. One
degree of freedom is used up by estimating the mean.
