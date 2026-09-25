# Data Structures & Algorithms (DSA) Solutions

This directory contains clean, benchmarked Python solutions to foundational interview problems with time/space complexity analysis.

---

## 📌 Problem Index

| # | Problem | Difficulty | Core Concept | Time Complexity | Space Complexity | Solution File |
|---|---|---|---|---|---|---|
| **217** | [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/) | Easy | Hash Set / $O(1)$ amortized lookup | $O(N)$ | $O(N)$ | [`leetcode_217_contains_duplicate.py`](leetcode_217_contains_duplicate.py) |
| **242** | [Valid Anagram](https://leetcode.com/problems/valid-anagram/) | Easy | Frequency Hash Map / Fixed Array mapping | $O(N)$ | $O(1)$ | [`leetcode_242_valid_anagram.py`](leetcode_242_valid_anagram.py) |

---

## 1. LeetCode 217: Contains Duplicate

### Core Concept
- **Hash Set**: By inserting items into a `set`, lookup is $O(1)$ amortized. As soon as an element is seen twice, we short-circuit and return `True`.
- **Contrast with Brute Force**: Comparing each element to all other elements takes $O(N^2)$ time with $O(1)$ space.
- **Contrast with Sorting**: Sorting takes $O(N \log N)$ time and $O(1)$ extra space, but mutates the input array or allocates a copy.

### Verification
Run tests locally:
```bash
python dsa/leetcode_217_contains_duplicate.py
```

---

## 2. LeetCode 242: Valid Anagram

### Core Concept
- **Frequency Counter**: Two strings are anagrams if and only if every character appears with the exact same frequency in both strings.
- **Approaches**:
  1. `collections.Counter`: Concise, idiomatic Python.
  2. Single hash map with increment/decrement: Generalizes to Unicode, terminates early if count drops below zero.
  3. Fixed 26-element array: For lowercase English strings (`ord(c) - ord('a')`), using strictly $O(1)$ space with zero hash table overhead.

### Verification
Run tests locally:
```bash
python dsa/leetcode_242_valid_anagram.py
```
