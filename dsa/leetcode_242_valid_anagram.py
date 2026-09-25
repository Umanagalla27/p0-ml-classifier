"""
LeetCode 242: Valid Anagram (Easy)
https://leetcode.com/problems/valid-anagram/

Given two strings s and t, return true if t is an anagram of s, and false otherwise.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase,
typically using all the original letters exactly once.

Complexity:
- Time Complexity: O(N) — N is length of strings s and t (single/double pass).
- Space Complexity: O(1) or O(K) — K is charset size (26 letters for English).

Alternative Approaches:
- Sorting: Sort both strings and compare: sorted(s) == sorted(t). Time: O(N log N).
- Frequency Array (size 26): Fixed-size integer array to count character balances.
- Hash Map (Counter or dict): Generalizes well to unicode. Time: O(N), Space: O(K).
"""

from collections import Counter


class Solution:
    def isAnagram_counter(self, s: str, t: str) -> bool:
        """Approach 1: Frequency Hash Map using collections.Counter."""
        if len(s) != len(t):
            return False
        return Counter(s) == Counter(t)

    def isAnagram_dict(self, s: str, t: str) -> bool:
        """Approach 2: Manual frequency map with single dictionary."""
        if len(s) != len(t):
            return False

        counts: dict[str, int] = {}
        for char in s:
            counts[char] = counts.get(char, 0) + 1

        for char in t:
            if char not in counts or counts[char] == 0:
                return False
            counts[char] -= 1

        return True

    def isAnagram_array(self, s: str, t: str) -> bool:
        """Approach 3: Fixed-size 26-element array for lowercase English letters."""
        if len(s) != len(t):
            return False

        char_counts = [0] * 26
        base = ord("a")

        for ch_s, ch_t in zip(s, t):
            char_counts[ord(ch_s) - base] += 1
            char_counts[ord(ch_t) - base] -= 1

        return all(count == 0 for count in char_counts)


# Quick unit test verification
if __name__ == "__main__":
    sol = Solution()

    test_cases = [
        ("anagram", "nagaram", True),
        ("rat", "car", False),
        ("a", "ab", False),
        ("ab", "a", False),
        ("", "", True),
        ("listen", "silent", True),
        ("fluster", "restful", True),
    ]

    for s, t, expected in test_cases:
        assert sol.isAnagram_counter(s, t) == expected, f"Failed counter on ({s}, {t})"
        assert sol.isAnagram_dict(s, t) == expected, f"Failed dict on ({s}, {t})"
        assert sol.isAnagram_array(s, t) == expected, f"Failed array on ({s}, {t})"

    print("All LeetCode 242 test cases passed!")
