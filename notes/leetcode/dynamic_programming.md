# Dynamic programming patterns (LeetCode) — SAMPLE NOTES

## When is it DP?
Look for two signs:
1. Optimal substructure: the best answer is built from best answers to
   smaller subproblems.
2. Overlapping subproblems: plain recursion would solve the same subproblem
   many times.
Phrases like "number of ways", "minimum cost", "longest", "can you reach" are
hints. If a greedy choice can be wrong, try DP.

## Memoization vs tabulation
- Memoization (top-down): write the recursion, cache results in a dict or
  @lru_cache. Easy to write from the recurrence. Only computes states you need.
- Tabulation (bottom-up): fill an array from the base cases up. No recursion
  depth limit, and easier to reduce memory (often to O(1) for 1D problems).
Start with memoization to get it correct, convert to tabulation if recursion
depth or memory is a problem.

## Defining the state
Ask: "what is the smallest description of a subproblem?" For coin change,
dp[a] = fewest coins to make amount a. Transition:
dp[a] = min(dp[a - c] + 1 for c in coins if c <= a), dp[0] = 0.

## House robber
dp[i] = max(dp[i-1], dp[i-2] + nums[i]). At house i you either skip it (keep
dp[i-1]) or rob it, which means you could not rob i-1 (dp[i-2] + nums[i]).
The max picks the better of the two choices.
