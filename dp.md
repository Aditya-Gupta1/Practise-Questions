
.# DP Questions

## 1. Given a set of non-negative integers, and a value sum, determine if there is a subset of the given set with sum equal to given sum.
```python

# def solve(arr, sums, n, dp):
# 	#Base Condition
# 	if sums == 0:
# 		return True
# 	elif n == 0:
# 		return False

# 	#Memoization
# 	if dp[n][sums] != 0:
# 		return dp[n][sums]
# 	if arr[n-1] <= sums:
# 		dp[n][sums] = (solve(arr, sums-arr[n-1], n-1, dp) or solve(arr, sums, n-1, dp))
# 		return dp[n][sums]
# 	else:
# 		dp[n][sums] = solve(arr, sums, n-1, dp)
# 		return dp[n][sums]

n = int(input())
arr = list(map(int, input().split()))
sums = int(input())

dp = [[0 for j in range(sums+1)] for i in range(n+1)]
for i in range(sums+1):
	dp[0][i] = False
for i in range(n+1):
	dp[i][0] = True

for i in range(1, n+1):
	for j in range(1, sums+1):
		if arr[i-1] <= j:
			dp[i][j] = (dp[i-1][j - arr[i-1]] or dp[i-1][j])
		else:
			dp[i][j] = dp[i-1][j]
print(dp[-1][-1])
```

## 2. Equal set partition problem: Divide the given array into two partition having same sum.
```python
As the 2 partitions must have same sum, so the total sum of array should be even. If it not even, then partition is not possible. If the sum is even, then call the subset sum function given above with sum = sum/2. Hence, if we can find a partition with half the sum, then there wil automatically exist other partition of the array.
```

## 3. Minimum Difference between 2 subsets. Given an array, the task is to divide it into two sets S1 and S2 such that the absolute difference between their sums is minimum.
```python
n = int(input())
arr = list(map(int, input().split()))
s = sum(arr)

dp = [[-1 for j in range(s+1)] for i in range(n+1)]

for j in range(s+1):
    dp[0][j] = False
for i in range(n+1):
    dp[i][0] = True
    
for i in range(1, n+1):
    for j in range(1, s+1):
        if arr[i-1] <= j:
            dp[i][j] = dp[i-1][j-arr[i-1]] or dp[i-1][j]
        else:
            dp[i][j] = dp[i-1][j]

res = 100000000
for j in range(s//2,-1,-1):
    if dp[-1][j] == True:
        res = s - (2*j)
        break
        
print(res)
```
## 4. Unbounded Knapsack. Given weights and values related to n items and the maximum capacity allowed for these items. What is the maximum value we can achieve if we can pick any weights any number of times for a total allowed weight of W?
```python
def unbounded_knapsack(val, wt, n, w, dp):
    
    for j in range(w+1):
        dp[0][j] = 0
    
    for i in range(n+1):
        dp[i][0] = 0
        
    for i in range(1, n+1):
        for j in range(1, w+1):
            if wt[i-1]<=j:
                dp[i][j] = max(dp[i-1][j], val[i-1]+dp[i][j- wt[i-1]])
            else:
                dp[i][j] = dp[i-1][j]
                

t = int(input())
for _ in range(t):
    n, w = list(map(int, input().split()))
    val = list(map(int, input().split()))
    wt = list(map(int, input().split()))
    
    dp = [[-1 for j in range(w+1)] for i in range(n+1)]
    
    unbounded_knapsack(val, wt, n, w, dp)
    
    print(dp[-1][-1])
```
## 5. Rod Cutting Problem. Given a rod of length n inches and an array of prices that contains prices of all pieces of size smaller than n. Determine the maximum value obtainable by cutting up the rod and selling the pieces.
```python
# Exactly Same as UNBOUNDED KNAPSACK PROBLEM.
def unbounded_knapsack(val, wt, n, dp):
    
    for j in range(n+1):
        dp[0][j] = 0
    
    for i in range(n+1):
        dp[i][0] = 0
        
    for i in range(1, n+1):
        for j in range(1, n+1):
            if wt[i-1]<=j:
                dp[i][j] = max(dp[i-1][j], val[i-1]+dp[i][j- wt[i-1]])
            else:
                dp[i][j] = dp[i-1][j]
                

t = int(input())
for _ in range(t):
    n = int(input())
    val = list(map(int, input().split()))
    wt = [i for i in range(1,n+1)]
    
    dp = [[-1 for j in range(n+1)] for i in range(n+1)]
    
    unbounded_knapsack(val, wt, n, dp)
    
    print(dp[-1][-1])
```
## 6. Coin Change Problem. Given a value N, find the number of ways to make change for N cents, if we have infinite supply of each of S = { S1, S2, .. , Sm} valued coins. The order of coins doesn’t matter. For example, for N = 4 and S = {1,2,3}, there are four solutions: {1,1,1,1},{1,1,2},{2,2},{1,3}. So output should be 4. For N = 10 and S = {2, 5, 3, 6}, there are five solutions: {2,2,2,2,2}, {2,2,3,3}, {2,2,6}, {2,3,5} and {5,5}. So the output should be 5.
```python
t = int(input())
for _ in range(t):
    n = int(input())
    coins = list(map(int, input().split()))
    s = int(input())
    dp = [[-1 for j in range(s+1)] for i in range(n+1)]
    for j in range(s+1):
        dp[0][j] = 0
    for i in range(n+1):
        dp[i][0] = 1
    for i in range(1, n+1):
        for j in range(1, s+1):
            if coins[i-1] <= j:
                dp[i][j] = dp[i-1][j] + dp[i][j - coins[i-1]]
            else:
                dp[i][j] = dp[i-1][j]
    print(dp[-1][-1])
```
## 7. Coin Change Problem - 2. Given a value V. You have to make change for V cents, given that you have infinite supply of each of C{ C1, C2, .. , Cm} valued coins. Find the minimum number of coins to make the change. Return -1 if not possible.
```python
def coinChange(self, coins: List[int], s: int) -> int:
        n = len(coins)
        
        dp = [[-1 for j in range(s+1)] for i in range(n+1)]
    
        for i in range(n+1):
            dp[i][0] = 0
        for j in range(s+1):
            dp[0][j] = 100000000

        for i in range(1, n+1):
            for j in range(1, s+1):
                if coins[i-1] <= j:
                    dp[i][j] = min(1+dp[i][j-coins[i-1]], dp[i-1][j])
                else:
                    dp[i][j] = dp[i-1][j]
                    
        if dp[-1][-1] == 100000000:
            return -1
        else:
            return dp[-1][-1]
```

## 8. Longest Common Subsequence. Given two sequences, find the length of longest subsequence present in both of them. Both the strings are of uppercase.
```python
t = int(input())
for _ in range(t):
    n, m = list(map(int, input().split()))
    s1 = input()
    s2 = input()
    dp = [[-1 for j in range(m+1)] for i in range(n+1)]
    for i in range(n+1):
        dp[i][0] = 0
    for j in range(m+1):
        dp[0][j] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = 1+dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
    print(dp[-1][-1])
```
## 9. Longest Common Substring.
```python
def longestCommonSubstring(a,b):
    m= len(a)
    n= len(b)
    dp = [[0 for j in range(n+1)] for i in range(m+1)]
    
    for i in range(m+1):
        dp[i][0]= 0
    for j in range(n+1):
        dp[0][j]= 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j]= dp[i-1][j-1]+1
            else:
                dp[i][j]= 0
                
    ans = 0
    for i in range(m+1):
        ans = max(ans, max(dp[i]))
    return ans
    
t = int(input())
for _ in range(t):
    m, n = list(map(int, input().split()))
    a = input()
    b = input()
    
    print(longestCommonSubstring(a,b))
```
## 9. Shortest Common Supersequence
```python
def LCS(a, b, m, n, dp):
    for i in range(m+1):
        dp[i][0]= 0
    for j in range(n+1):
        dp[0][j]= 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j]= dp[i-1][j-1]+1
            else:
                dp[i][j]= max(dp[i][j-1], dp[i-1][j])
    return dp[-1][-1]

t = int(input())
for _ in range(t):
    a, b = input().split()
    m = len(a)
    n = len(b)
    dp = [[-1 for j in range(n+1)] for i in range(m+1)]
    l = LCS(a, b, m, n, dp)
    print(m+n-l)
```
## 10. Minimum Number of deletions and Insertions.
```python
def lcs(a, b):
    m = len(a)
    n = len(b)
    dp = [[0 for j in range(n+1)] for i in range(m+1)]
    
    for i in range(m+1):
        dp[i][0] = 0
    for j in range(n+1):
        dp[0][j] = 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = 1+dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]
    
t = int(input())
for _ in range(t):
    m, n = list(map(int, input().split()))
    a, b = input().split()
    l = lcs(a, b)
    ins = m - l
    dele = n - l
    print(ins+dele)
```
## 11. Longest Palindromic Subsequence.
```python
def lcs(a,b):
    m = len(a)
    n = len(b)
    dp = [[0 for j in range(n+1)] for i in range(m+1)]
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = 1+dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]
def longestPalindromeSubseq(s: str) -> int:
        return lcs(s, s[::-1])
```
## 12. Minimum Number of deletions to make a string palindrome.
```python
# This exact code can be used to find the minimum number of insertions to make a string palindrome.
def lcs(a,b):
    m = len(a)
    n = len(b)
    dp = [[0 for j in range(n+1)] for i in range(m+1)]
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = 1+dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
    return dp[-1][-1]
def minInsertions(self, s: str) -> int:
        return len(s) - lcs(s, s[::-1])
```
## 13. Longest repeating subsequence.
```python
def lrs(s, n):
    dp = [[0 for j in range(n+1)] for i in range(n+1)]
    for i in range(1,n+1):
        for j in range(1, n+1):
            if s[i-1] == s[j-1] and i!=j:
                dp[i][j] = 1+dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i][j-1], dp[i-1][j])
                
    return dp[-1][-1]
```
## 14. 