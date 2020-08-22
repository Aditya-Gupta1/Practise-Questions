# LeetCode Questions

## 1. Path Sum III. You are given a binary tree in which each node contains an integer value. Find the number of paths that sum to a given value. The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).
```python
class Solution:
    
    def pathSum(self, root: TreeNode, sum: int) -> int: ## Main function
        
        self.final_ans = 0
        self.dfs(root, sum)
        
        return self.final_ans
    
    def dfs(self, root, target):
        
        if root is None:
            return
        
        self.findPath(root, target)
        self.dfs(root.left, target)
        self.dfs(root.right, target)
        
    def findPath(self, root, target):
        if root is None:
            return
        if root.val == target:
            self.final_ans += 1
            
        self.findPath(root.left, target - root.val)
        self.findPath(root.right, target - root.val)

```

## 2. Rotting Oranges. In a given grid, each cell can have one of three values:

    the value 0 representing an empty cell;
    the value 1 representing a fresh orange;
    the value 2 representing a rotten orange.

Every minute, any fresh orange that is adjacent (4-directionally) to a rotten orange becomes rotten. Return the minimum number of minutes that must elapse until no cell has a fresh orange.  If this is impossible, return -1 instead.
```python
def orangesRotting(arr: List[List[int]]) -> int:
        row, col = len(arr), len(arr[0])
        rotting = {(i, j) for i in range(row) for j in range(col) if arr[i][j] == 2}
        fresh = {(i, j) for i in range(row) for j in range(col) if arr[i][j] == 1}
        timer = 0
        while fresh:
            if not rotting:
                return -1
            rotting = {(i+di, j+dj) for i, j in rotting for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)] if (i+di, j+dj) in fresh}
            fresh -= rotting
            timer += 1
            
        return timer
```
## 3. Excel Sheet Column Number.
```python
def titleToNumber(self, s: str) -> int:
        
        char_dict = {}
        characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        x = 1
        for ch in characters:
            char_dict[ch] = x
            x += 1
            
        n = len(s)
        ans = 0
        
        for i in range(n):
            ans += 26**(n-i-1)*char_dict[s[i]]
            
        return ans
```
## 4. H-Index.
```python
def hIndex(self, citations: List[int]) -> int:
        return sum([ (j > i) for i, j in enumerate(sorted(citations, reverse= True))])
```
## 5. Iterator for combination.
```python
from itertools import combinations

class CombinationIterator:

    def __init__(self, characters: str, combinationLength: int):
        self.i = 0
        self.input = list(characters)
        self.c = combinations(self.input, combinationLength)
        x = []
        for i in list(self.c):
            x.append(''.join(i))
        self.c = x
        
        

    def next(self) -> str:
        x = self.c[self.i]
        self.i += 1
        return x
        

    def hasNext(self) -> bool:
        if self.i == len(self.c):
            return False
        else:
            return True
```
## 6. Longest Palindrome.
```python
def longestPalindrome(self, s: str) -> int:
        freq = {}
        for i in s:
            freq[i] = freq.get(i, 0) + 1
        res = 0
        for k, v in freq.items():
            res += (v//2)*2
            if res%2 == 0 and v%2 == 1:
                res += 1
        return res
```
## 7. Non-OverLapping Intervals.
```python
def eraseOverlapIntervals(self, arr: List[List[int]]) -> int:
        if len(arr) == 0:
            return 0
        arr.sort()
        n = len(arr)
        forward_len = 1
        backward_len = 1
        f_e = arr[0][1]
        b_e = arr[-1][0]
        
        for i in range(1, n):
            if f_e <= arr[i][0]:
                f_e = arr[i][1]
                forward_len += 1
        
        for j in range(n-2, -1, -1):
            if b_e >= arr[j][1]:
                b_e = arr[j][0]
                backward_len += 1
        
        f_l = n - forward_len
        b_l = n - backward_len
        
        return min(f_l, b_l)
```

## 8. Best time to Buy and Sell Stock III
```python
def maxProfit(self, arr: List[int]) -> int:
        n = len(arr)
        if n == 0:
            return 0
        left = [0]*n
        minv = arr[0]
        for i in range(1, n):
            if arr[i] < minv:
                minv = arr[i]
                left[i] = left[i-1]
            else:
                left[i] = max(left[i-1], arr[i] - minv)
        right = [0]*n
        maxv = arr[n-1]
        for i in range(n-2, -1, -1):
            if arr[i] > maxv:
                maxv = arr[i]
                right[i] = right[i+1]
            else:
                right[i] = max(right[i+1], maxv - arr[i])
                
        ans = 0
        for i in range(n):
            ans = max(ans, right[i]+left[i])
        
        return ans
```
## 9. Distribute Candies to People
```python
def distributeCandies(self, candies: int, n: int) -> List[int]:
        res = [0]*n
        rem = candies
        c = 1
        i = 0
        while True:
            
            if rem - c <= 0:
                res[i] += rem
                break
            res[i] += c
            rem -= c
            c = c+1
            i = (i+1)%n
            
        return res
```
## 10. Numbers with same consecutive differences.
```python
def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        ans = []
        if n == 1:
            return [0,1,2,3,4,5,6,7,8,9]
        def solve(s,k,i,n):
            if len(s) == n:
                ans.append(s)
                return
            if i > 9:
                return
            if i+k < 10:
                solve(s+str(i), k, i+k, n)
            if i-k > -1:
                solve(s+str(i), k, i-k, n)
                
        for i in range(1, 10):
            solve("",k,i,n)
        
        return list(set(ans))
```
## 11. Goat Latin
```python
def toGoatLatin(self, s: str) -> str:
        n = len(s)
        vowels = ['a','e','i','o','u','A','E','I','O','U']
        
        res = []
        s += ' '
        w = ""
        c = 1
        for i in range(n+1):
            if s[i]!=' ':
                w += s[i]
            else:
                print('else',w)
                if w[0] in vowels:
                    w += 'ma'
                else:
                    
                    if len(w) == 1:
                        w += 'ma'
                    else:
                        w = w[1:]+w[0]+'ma'
                w += 'a'*(c)
                c += 1
                res.append(w)
                w = ''
        print(res)
        return ' '.join(res)
```
## 12. Reorder List
```python
def reverse(root):
    
    curr = root
    prev = None
    while curr:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next
    return prev
def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if head == None:
            return
        fast = slow = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        sec_half = slow.next
        slow.next = None
        first_half = head
        
        sec_half = reverse(sec_half)
        
        while first_half and sec_half:
            p = first_half.next
            first_half.next = sec_half
            q = sec_half.next
            sec_half.next = p
            first_half = p
            sec_half = q
```
## 13. Sort Array by Parity
```python
def sortArrayByParity(self, arr: List[int]) -> List[int]:
        n = len(arr)
        pointer = -1
        for i in range(len(arr)):
            if arr[i]%2 == 0:
                pointer += 1
                arr[i], arr[pointer] = arr[pointer], arr[i]
                
        return arr
```
## 14. 