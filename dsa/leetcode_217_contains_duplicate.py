"""
LeetCode 217: Contains Duplicate (Easy)
https://leetcode.com/problems/contains-duplicate/

Given an integer array nums, return true if any value appears at least twice in the array,
and return false if every element is distinct.

Complexity:
- Time Complexity: O(N) — Single pass through the array with O(1) average lookup/insertion.
- Space Complexity: O(N) — In the worst case (all elements unique), the hash set stores N elements.

Alternative Approaches:
- Brute Force: Check every pair (i, j). Time: O(N^2), Space: O(1).
- Sorting: Sort the array and check adjacent elements. Time: O(N log N), Space: O(1) or O(N) depending on sort.
"""

from typing import List


class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        """Determines if any integer appears at least twice using a hash set."""
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

    def containsDuplicate_one_liner(self, nums: List[int]) -> bool:
        """Pythonic one-liner comparing length of list vs length of set."""
        return len(nums) != len(set(nums))


# Quick unit test verification
if __name__ == "__main__":
    sol = Solution()

    # Test cases
    assert sol.containsDuplicate([1, 2, 3, 1]) is True
    assert sol.containsDuplicate([1, 2, 3, 4]) is False
    assert sol.containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]) is True
    assert sol.containsDuplicate([]) is False
    assert sol.containsDuplicate([42]) is False

    assert sol.containsDuplicate_one_liner([1, 2, 3, 1]) is True
    assert sol.containsDuplicate_one_liner([1, 2, 3, 4]) is False

    print("All LeetCode 217 test cases passed!")
