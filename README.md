# Practise Questions

## 1. Check whether 2 strings are anagrams or not.
```python
strings = input().split()
if "".join(sorted(strings[0])) == "".join(sorted(strings[1])):
   print('Anagrams')
else:
  print('No Anangrams')
```
## 2. Check whether 2 strings are palindrome or not.
```python
s = input()
if s == s[-1::-1]:
  print('Palindrome')
else:
  print('Not a Palindrome')
```
## 3. Remove all consecutive duplicates from a given string.
```python
s = input()
l = []
for i in range(len(s)-1):
    if s[i]!=s[i+1]:
        l.append(s[i])
if s[-2] != s[-1]:
    l.append(s[-1])
print("".join(l))
```
## 4. Return the second most repeated character from the string.
```python
s = input()
d = dict()
for i in s:
    if i in d:
        d[i] +=1
    else:
        d[i] = 1
l = sorted(d.items(), key= lambda x:(x[1],x[0]))
print(l[-1][0])
```
## 5. Return common characters of two strings in alphabetical order.
```python
s1,s2 = input().split()
l = []
for i in s1:
    if i in s2:
        l.append(i)
print(l)
```
## 6. Add two bit strings.
```python
s1,s2 = input().split()
x = int(s1,2)+int(s2,2)
print('Decimal Equivalent :',x)
ans = str(bin(x))
ans = ans[2:]
print('Binary Equivalent :',ans)
```
## 7. Given a string, find the longest length of a prefix which is also a suffix.
**Hint :** This question refers to the computelps method of KMP String Matching Algorithm. See GeeksforGeeks for details.
```python
p = input()
m = len(p)
l = [0 for i in range(m)]
maxlen = 0
i = 1
while i < m:
    if p[i] == p[maxlen]:
        maxlen += 1
        l[i] = maxlen
        i += 1
    else:
        if maxlen != 0:
            maxlen = l[maxlen - 1]
        else:
            l[i] = 0
            i += 1
print(l)
```
## 8. KMP Pattern Matching Algorithm
```python
def KMPsearch(s,p):
    i = 0
    j = 0
    m = len(p)
    n = len(s)
    l = computeLPS(p)
    while i < n:
        if s[i] == p[j]:
            i += 1
            j += 1
        if j == m:
            print('Pattern found at index '+ str(i-j))
            j = l[j-1]
        elif i < n and p[j] != s[i]:
            if j != 0:
                j = l[j-1]
            else:
                i += 1
def computeLPS(p):
    m = len(p)
    l = [0 for i in range(m)]
    maxlen = 0
    i = 1
    while i < m:
        if p[i] == p[maxlen]:
            maxlen += 1
            l[i] = maxlen
            i += 1
        else:
            if maxlen != 0:
                maxlen = l[maxlen - 1]
            else:
                l[i] = 0
                i += 1
    return l;
KMPsearch('ABABDABACDABABCABAB','ABABCABAB')
```
## 9. Given a string of brackets (, ), {, }, [, ], find the position in the string where the orders of brackets breaks.
```python
s = input()
l = []
flag = -1
for ind , i in enumerate(s):
    if i =='(' or i == '{' or i == '[':
        l.append((i,ind))
    else:
        try:
            if ( i == ')' and l[-1][0] == '(') or (i == '}' and l[-1][0] == '{') or (i == ']' and l[-1][0] == '['):
                del l[-1]
            else:
                flag = ind
        except:
            flag = ind
if len(l) == 1 and (l[0][0] == '(' or l[0][0] == '{' or l[0][0] == '['):
    flag = l[0][1]
if flag != -1:
    print(flag+1)
else:
    print(flag)
```
## 10. Given a list of string and numbers, rotate the string by one position to the right if the sum of squares of digits of the corresponding number is even and rotate it twice to the left if the sum of the squares of the digits of the corresponding number is odd.
```python
l1= input().split(' ')
l2 = list(map(int,input().split(' ')))
def sumsquares(n):
    sum = 0
    while n > 0:
        sum += (n % 10)**2
        n = n //10
    return sum
ans = []
for x,y in zip(l1,l2):
    if sumsquares(y)%2 == 0:
        x = x[-1]+x[:-1]
    else:
        x = x[2:]+x[:2]
    ans.append(x)
print(ans)
```
## 11. Given an alphanumeric string, extract all numbers, remove the duplicate digits, and from that set of digits construct the largest even number possible
```python
def extractnumbers(s):
    ans = ''
    for i in s:
        if i.isalpha() == False:
            ans += i
    return ans
s = extractnumbers(input())
s = ''.join(dict.fromkeys(s))
s = sorted(s,reverse=True)
if int(s[-1])%2 != 0:
    for i in range(len(s)-2,0,-1):
        if int(s[i])%2 ==0:
            x = s[-1]
            s[-1] = s[i]
            s[i] = x
            break
print(''.join(s))
```
## 12. Find the length of largest subarray with 0 sum.
```python
def maxLen(n, lis):
    d = {}
    sum = 0
    maxl = 0
    for i in range(len(lis)):
        sum += lis[i]
        if lis[i] == 0 and maxl == 0:
            maxl = 1
        if sum == 0:
            maxl = i+1
        if sum in d:
            maxl = max(maxl,i - d[sum])
        else:
            d[sum] = i
    return maxl
```
## 13. Largest subarray of 0's and 1's.
```python
def maxLen(n, lis):
    for i in range(len(lis)):
        if lis[i] == 0:
            lis[i] = -1
    d = {}
    sum = 0
    maxl = 0
    for i in range(len(lis)):
        sum += lis[i]
        if lis[i] == 0 and maxl == 0:
            maxl = 1
        if sum == 0:
            maxl = i+1
        if sum in d:
            maxl = max(maxl,i - d[sum])
        else:
            d[sum] = i
    return maxl
```
## 14. Subarray with given sum .
```python
t = int(input())
for _ in range(t):
    n, s = list(map(int,input().split()))
    l = list(map(int,input().split()))
    #using sliding window approach
    start = 0
    last = 0
    sum = 0
    flag = 0
    for i in range(len(l)):
        sum += l[i]
        last = i+1
        if sum == s:
            flag = 1
            print('{} {}'.format(start+1,last))
            break
        if sum > s:
            while sum > s and start < len(l):
                sum -= l[start]
                if sum == s:
                    print('{} {}'.format(start+2,last))
                    flag = 1
                    break
                start += 1
        if flag == 1:
            break
    if flag == 0:
        print(-1)
```
## 15. Sort an array of 0s, 1s and 2s.
```python
t = int(input())
for _ in range(t):
    n = int(input())
    l = list(map(int,input().split()))
    low = 0
    mid = 0
    high = len(l)-1
    while(mid <= high):
        if l[mid] == 0:
            l[low],l[mid] = l[mid],l[low]
            low += 1
            mid += 1
        elif l[mid] == 2:
            l[mid],l[high] = l[high],l[mid]
            high -= 1
        else:
            mid += 1
    for x in l:
        print(x,end=' ')
    print()
```
## 16. Given two arrays X and Y of positive integers, find number of pairs such that x^y > y^x (raised to power of) where x is an element from X and y is an element from Y.
```python
import bisect
def count(x,Y,n,counts):
    if x == 0:
        return 0
    if x == 1:
        return counts[0]
    idx = bisect.bisect_right(Y,x)
    ans = n - idx
    ans += (counts[0]+counts[1])
    if x == 2:
        ans -= (counts[3]+counts[4])
    if x == 3:
        ans += counts[2]
    return ans
def countPairs(X,Y,m,n):
    counts = [0]*5
    for y in Y:
        if y < 5:
            counts[y] += 1
    Y.sort()
    totalPairs = 0
    for x in X:
        totalPairs += count(x,Y,n,counts)
    return totalPairs

t = int(input())
for _ in range(t):
    m, n= list(map(int,input().split()))
    X = list(map(int,input().split()))
    Y = list(map(int,input().split()))
    print(countPairs(X,Y,m,n))
```
## 17. Given an array of positive integers. The task is to find inversion count of array.Inversion Count : For an array, inversion count indicates how far (or close) the array is from being sorted. If array is already sorted then inversion count is 0. If array is sorted in reverse order that inversion count is the maximum. Formally, two elements a[i] and a[j] form an inversion if a[i] > a[j] and i < j.
```python
def mergeSort(arr,res,l,r):
    count = 0
    if l<r:
        m = (l +r)//2
        count += mergeSort(arr,res,l,m)
        count += mergeSort(arr,res,m+1,r)
        count += merge(arr,res,l,m,r)
    return count
def merge(arr,res,l,m,r):
    i = l
    j = m+1
    k = l
    inv_count = 0
    while i <= m and j <= r:
        if arr[i]<=arr[j]:
            res[k] = arr[i]
            k+=1
            i+=1
        else:
            res[k] = arr[j]
            inv_count += (m - i + 1)
            k+=1
            j+=1
    while i <= m:
        res[k] = arr[i]
        i+=1
        k+=1
    while j <= r:
        res[k] = arr[j]
        j+=1
        k+=1
    for x in range(l,r+1):
        arr[x] = res[x]
    return inv_count
t = int(input())
for _ in range(t):
    n = int(input())
    l = list(map(int,input().split()))
    res = [0]*n
    print(mergeSort(l,res,0,n-1))
```
## 18. Kadane's Algorithm. Given an array arr of N integers. Find the contiguous sub-array with maximum sum.
```python
t = int(input())
for _ in range(t):
    n = int(input())
    l = list(map(int,input().split()))
    max_so_far = l[0]
    max_ending_here = l[0]
    for x in range(1,len(l)):
        max_ending_here = max(max_ending_here+l[x],l[x])
        max_so_far = max(max_so_far,max_ending_here)
    print(max_so_far)
```
## 19. The cost of stock on each day is given in an array A[] of size N. Find all the days on which you buy and sell the stock so that in between those days your profit is maximum.<br>
Input:<br>
2<br>
7<br>
100 180 260 310 40 535 695<br>
10<br>
23 13 25 29 33 19 34 45 65 67<br>
Output:<br>
(0 3) (4 6)<br>
(1 4) (5 9)<br>
```python
t = int(input())
for _ in range(t):
    n = int(input())
    l = list(map(int,input().split()))
    flag = 0
    x = []
    for i in range(n-1):
        if l[i]<=l[i+1] and flag == 0:
            x.append(i)
            flag = 1
        elif l[i]>l[i+1] and flag == 1:
            x.append(i)
            flag = 0
        if l[i]<=l[i+1] and flag == 1 and i == n-2:
            x.append(i+1)
    if len(x) == 0:
        print('No Profit',end=" ")
    else:
        for i in range(0,len(x),2):
            print('({} {})'.format(x[i],x[i+1]),end=' ')
    print()
```
## 20. Given a string S and text T. Output the smallest window in the string S having all characters of the text T. Both the string S and text T contains lowercase english alphabets.
```python
t = int(input())
for _ in range(t):
    s = input()
    text = input()
    len1 = len(s)
    len2 = len(text)
    s_map = [0]*256
    p_map = [0]*256
    minLength = float('inf')
    start = 0
    start_index = -1
    count = 0
    for i in range(len2):
        p_map[ord(text[i])] += 1
    for i in range(len1):
        s_map[ord(s[i])] += 1
        if p_map[ord(s[i])] != 0 and s_map[ord(s[i])] <= p_map[ord(s[i])]:
            count += 1
        if count == len2:
            while s_map[ord(s[start])] > p_map[ord(s[start])] or p_map[ord(s[start])] == 0:
                if s_map[ord(s[start])] > p_map[ord(s[start])]:
                    s_map[ord(s[start])] -= 1
                start +=1
            window = i - start + 1
            if minLength > window:
                minLength = window
                start_index = start
    if start_index == -1:
        print(-1)
    else:
        print(s[start_index:start_index+minLength])
```
## 21. Given an array if ‘n’ positive integers, count number of pairs of integers in the array that have the sum divisible by k. 
```python
t = int(input())
for _ in range(t):
    n,k = list(map(int,input().split()))
    l = list(map(int,input().split()))
    freq = [0]*k
    for i in range(n):
        freq[l[i]%k] += 1
    sum = freq[0]*(freq[0]-1)/2
    i = 1
    while i <= k//2 and i != k-i:
        sum += freq[i]*freq[k-i]
        i += 1
    if k%2 == 0:
        sum += freq[k//2]*(freq[k//2]-1)/2
    print(int(sum))
```
## 21. BFS implementation in Python
```python
from collections import defaultdict
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    def addEdge(self,src,dest):
        self.graph[src].append(dest)
        self.graph[dest].append(src)
    def bfs(self,s):
        visited = [False]*len(self.graph)
        q = []
        q.append(s)
        visited[s] = True
        while q:
            s = q.pop(0)
            print(s,end=' ')
            for i in self.graph[s]:
                if visited[i] == False:
                    q.append(i)
                    visited[i] = True
g = Graph()
g.addEdge(0, 1) 
g.addEdge(0, 2) 
g.addEdge(1, 2) 
g.addEdge(2, 0) 
g.addEdge(2, 3) 
g.addEdge(3, 3) 
g.bfs(2)
```
## 22. Connected Components in a Graph.
```python
def dfsUtil(self,k,visited,l):
        visited[k] = True
        l.append(k)
        for i in self.graph[k]:
            if not visited[i]:
                visited[i] = True
                self.dfsUtil(i,visited,l)
    def connectedcomponents(self):
        visited = [False]*len(self.graph)
        for k in self.graph.keys():
            if not visited[k]:
                l = []
                self.dfsUtil(k,visited,l)
                print(l)
```
## 23. Given an array arr[] and a number K where K is smaller than size of array, the task is to find the Kth smallest element in the given array. It is given that all array elements are distinct.
Expected Time Complexity: O(n)
``` python
from random import randint
def partition(arr,low,high): 
    i = ( low-1 )
    pivot = arr[high]
    for j in range(low , high):
        if   arr[j] < pivot:
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i]
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return i+1
t = int(input())
for _ in range(t):
    n = int(input())
    arr = list(map(int,input().split()))
    k = int(input())
    l = 0
    r = n-1
    while True:
        x = randint(l,r)
        arr[x],arr[r] = arr[r],arr[x]
        p = partition(arr,l,r)
        if p == k-1:
            print(arr[p])
            break
        elif p > k-1:
            r = p
        else:
            l = p+1
```
## 24. Given arrival and departure times of all trains that reach a railway station. Your task is to find the minimum number of platforms required for the railway station so that no train waits. Arrival and departure times will not be same for a train, but we can have arrival time of one train equal to departure of the other. In such cases, we need different platforms, i.e at any given instance of time, same platform can not be used for both departure of a train and arrival of another.
```python
t = int(input())
for _ in range(t):
    n = int(input())
    arr = list(map(int,input().split()))
    dep = list(map(int,input().split()))
    arr.sort()
    dep.sort()
    platforms = 1
    result = 1
    i = 1
    j = 0
    while(i < n and j < n):
        if arr[i]<=dep[j]:
            platforms += 1
            i += 1
            if platforms > result:
                result = platforms
        else:
            platforms -= 1
            j += 1
    print(result)
```
## 25. Given an array arr[] of N non-negative integers representing height of blocks at index i as Ai where the width of each block is 1. Compute how much water can be trapped in between blocks after raining.
<img src='watertrap.png'></img>
```python
def trappingWater(arr,n):
    l = [0]*n
    r = [0]*n
    l[0] = arr[0]
    r[n-1] = arr[n-1]
    lmax = arr[0]
    rmax = arr[n-1]
    for i in range(1,n):
        if arr[i] > lmax:
            lmax = arr[i]
        l[i] = lmax
    for i in range(n-2,-1,-1):
        if arr[i]> rmax:
            rmax = arr[i]
        r[i] = rmax
    ans = 0
    for i in range(n):
        ans += min(l[i],r[i]) - arr[i]
    return ans
```
## 26. Given a boolean matrix mat[M][N] of size M X N, modify it such that if a matrix cell mat[i][j] is 1 (or true) then make all the cells of ith row and jth column as 1.
```python
t = int(input())
for _ in range(t):
    m,n = list(map(int,input().split()))
    arr = []
    for i in range(m):
        l = input().split()
        arr.append(l)
    row = [0]*m
    col = [0]*n
    for i in range(m):
        for j in range(n):
            if arr[i][j] == '1':
                row[i] = '1'
                col[j] = '1'
    for i in range(m):
        for j in range(n):
            if row[i] == '1' or col[j] == '1':
                arr[i][j] = '1'
    for i in range(m):
        for j in range(n):
            print(arr[i][j],end= ' ')
        print()
```
## 27. Given a number N, calculate the prime numbers upto N using Sieve of Eratosthenes.  
```python
def seive(n):
    arr = [True for i in range(n+1)]
    p = 2
    while(p*p <= n):
        if arr[p] == True:
            for i in range(p*p,n+1,p):
                arr[i] = False
        p+=1
    for x in range(2,n):
        if arr[x] == True:
            print(x,end= ' ')
    print()
t = int(input())
for _ in range(t):
    n = int(input())
    seive(n)
```
## 28. Given two sorted arrays arr1[] and arr2[] in non-decreasing order with size n and m. The task is to merge the two sorted arrays into one sorted array (in non-decreasing order).
```python
t = int(input())
for _ in range(t):
    x,y = list(map(int,input().split()))
    arr = list(map(int,input().split()))
    brr = list(map(int,input().split()))
    i = 0
    j = 0
    while(i < x and j < y):
        if arr[i]<= brr[j]:
            print(arr[i], end= ' ')
            i+=1
        else:
            print(brr[j],end= ' ')
            j += 1
    while i < x:
        print(arr[i],end=' ')
        i+=1
    while j < y:
        print(brr[j],end=' ')
        j += 1
    print()
```