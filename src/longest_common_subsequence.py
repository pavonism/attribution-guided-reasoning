def get_longest_common_word_subsequence_length(first: str, second: str) -> int:
    first = [word.lower() for word in first.split(" ")]
    second = [word.lower() for word in second.split(" ")]

    max_len = max(len(first), len(second))
    cache = [
        [0] * (max_len + 1),
        [0] * (max_len + 1),
    ]

    for i in range(max_len - 1, -1, -1):
        if i < len(first):
            f = i  # f indexes the first string
            cache[0][f] = cache[0][f + 1]  # Initialize as if first[f] would be skipped
            # Try to extend by matching first[f] with second[f:]
            for s in range(len(second) - 1, f - 1, -1):
                if first[f] == second[s]:
                    cache[0][f] = max(cache[0][f], 1 + cache[0][f + 1])

        if i < len(second):
            s = i  # s indexes the second string
            cache[1][s] = cache[1][s + 1]  # Initialize as if second[i] would be skipped
            # Try to extend by matching second[s] with first[s:]
            for f in range(len(first) - 1, s - 1, -1):
                if second[s] == first[f]:
                    cache[1][s] = max(cache[1][s], 1 + cache[1][s + 1])

    return max(cache[0][0], cache[1][0])


def get_longest_common_word_subsequence_length_memoization(
    first: str, second: str
) -> int:
    first = [word.lower() for word in first.split(" ")]
    second = [word.lower() for word in second.split(" ")]
    cache = {}

    def lcs(i: int, j: int) -> int:
        if i >= len(first) or j >= len(second):
            return 0
        if (i, j) not in cache:
            longest = 0
            if first[i] == second[j]:
                longest = 1 + lcs(i + 1, j + 1)
            cache[(i, j)] = max(
                longest,
                lcs(i + 1, j),
                lcs(i, j + 1),
            )
        return cache[(i, j)]

    return lcs(0, 0)
