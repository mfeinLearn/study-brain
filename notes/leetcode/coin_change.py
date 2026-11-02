"""LeetCode 322 Coin Change — SAMPLE NOTES.

Dynamic programming, bottom-up tabulation.
dp[a] = fewest coins needed to make amount a.
"""


def coin_change(coins: list[int], amount: int) -> int:
    INF = amount + 1
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] != INF else -1


# Top-down memoization version of the same recurrence.
from functools import lru_cache


def coin_change_memo(coins: list[int], amount: int) -> int:
    @lru_cache(maxsize=None)
    def best(a: int) -> float:
        if a == 0:
            return 0
        if a < 0:
            return float("inf")
        return min(best(a - c) + 1 for c in coins)

    ans = best(amount)
    return -1 if ans == float("inf") else int(ans)
