# Stacks

## 1. Stock Span Problem - GFG.
```python
def ngl(arr, n): #nearest greater to left
    stk = []
    ans = []
    
    for i in range(n):
        if len(stk) == 0:
            ans.append(-1)
        elif arr[stk[-1]] >= arr[i]:
            ans.append(stk[-1])
        else:
            while len(stk) != 0 and arr[stk[-1]] < arr[i]:
                del stk[-1]
                
            if len(stk) == 0:
                ans.append(-1)
            else:
                ans.append(stk[-1])
        stk.append(i)
        
    return ans
    
t = int(input())
for test in range(t):
    n = int(input())
    arr = list(map(int, input().split()))
    
    res = ngl(arr, n)
    
    out = [0]*n
    
    for i in range(n):
        out[i] = i - res[i]
        print(str(out[i])+' ',end='')
    if test != t-1:
        print()
```
## 2. Sort a Stack. Given a stack, the task is to sort it such that the top of the stack has the greatest element. Example: Input - [11, 2, 32, 3, 41] Output - [41, 32, 11, 3, 2].
```python
# Method -1: Using temporary array
def sorted(s):
    # Code here
    temp = []
    
    while len(s) != 0:
        x = s.pop()
        while len(temp) != 0 and temp[-1] < x:
            s.append(temp.pop())
        temp.append(x)
        
    s = temp
    print(temp)
    return s
# Method - 2: Using Recursion
def sinsert(s, x):
    if len(s) == 0 or s[-1] > x:
        s.append(x)
    else:
        temp = s.pop()
        sinsert(s, x)
        s.append(temp)

def sorted(s): # Main Function
    if len(s) != 0:
        x = s.pop()
        sorted(s)
        sinsert(s, x)
```
## 3. Maximum Area in a Histogram. Find the largest rectangular area possible in a given histogram where the largest rectangle can be made of a number of contiguous bars. For simplicity, assume that all bars have same width and the width is 1 unit.
```python
# Method-1
#code
def nsl(arr, n):
    
    stk = []
    ans = []
    
    for i in range(n):
        if len(stk) == 0:
            ans.append(-1)
        elif arr[stk[-1]] < arr[i]:
            ans.append(stk[-1])
        else:
            while len(stk) != 0 and arr[stk[-1]] >= arr[i]:
                stk.pop()
                
            if len(stk) == 0:
                ans.append(-1)
            else:
                ans.append(stk[-1])
        stk.append(i)
        
    return ans
    
def nsr(arr, n):
    
    stk = []
    ans = []
    
    for i in range(n-1, -1, -1):
        if len(stk) == 0:
            ans.append(n)
        elif arr[stk[-1]] < arr[i]:
            ans.append(stk[-1])
        else:
            while len(stk) != 0 and arr[stk[-1]] >= arr[i]:
                stk.pop()
                
            if len(stk) == 0:
                ans.append(n)
            else:
                ans.append(stk[-1])
        stk.append(i)
        
    return ans[::-1]
    
t = int(input())
for _ in range(t):
    n = int(input())
    arr = list(map(int, input().split()))
    
    right = nsr(arr, n)
    left = nsl(arr, n)
    width = [0]*n
    max_area = 0
    
    for i in range(n):
        width[i] = right[i] - left[i] - 1
        max_area = max(max_area, width[i]*arr[i])
    
    print(max_area)

# Method-2
def getMaxArea(arr):
    n = len(arr)
    
    stk = []
    
    i = 0
    max_area = 0
    
    while i < n:
        # print(stk)
        # print(i)
        if len(stk) == 0 or arr[stk[-1]] <= arr[i]:
            stk.append(i)
            i += 1
        else:
            top = stk.pop()
            
            area = arr[top] * ((i - stk[-1] - 1) if stk else i)
            
            max_area = max(max_area, area)
            
    while stk:
        top = stk.pop()
            
        area = arr[top] * ((i - stk[-1] - 1) if stk else i)
        
        max_area = max(max_area, area)
        
    return max_area
```

## 4. Max Area of a Rectngle in a Binary Matrix. Given a binary matrix. Find the maximum area of a rectangle formed only of 1s in the given matrix.
```python
def getMaxArea(arr):
    n = len(arr)
    
    stk = []
    
    i = 0
    max_area = 0
    
    while i < n:
        # print(stk)
        # print(i)
        if len(stk) == 0 or arr[stk[-1]] <= arr[i]:
            stk.append(i)
            i += 1
        else:
            top = stk.pop()
            
            area = arr[top] * ((i - stk[-1] - 1) if stk else i)
            
            max_area = max(max_area, area)
            
    while stk:
        top = stk.pop()
            
        area = arr[top] * ((i - stk[-1] - 1) if stk else i)
        
        max_area = max(max_area, area)
        
    return max_area


def maxRectangle(arr, r, c):
    
    max_area = 0
    temp = [0]*c
    
    for i in range(r):
        for j in range(c):
            if arr[i][j] == 0:
                temp[j] = 0
            else:
                temp[j] = temp[j]+arr[i][j]
        area = getMaxArea(temp)
        max_area = max(max_area, area)
    return max_area
```
## 5. Trapping Rain Water.
```java
//Method-1
public static int[] getLeft(int arr[], int n){
        int ans[] = new int[n];
        ans[0] = arr[0];
        for(int i = 1 ; i < n ; i ++)
        ans[i] = Math.max(ans[i-1], arr[i]);
        return ans;
    }
    public static int[] getRight(int arr[], int n){
        int ans[] = new int[n];
        ans[n-1] = arr[n-1];
        for(int i = n-2 ; i >= 0 ; i --)
        ans[i] = Math.max(ans[i+1], arr[i]);
        return ans;
    }
	public static void main (String[] args) {
		Scanner in = new Scanner(System.in);
		int t = in.nextInt();
		for(int r =0; r < t ; r ++){
		    int n = in.nextInt();
		    int arr[] = new int[n];
		    
		    for(int i = 0; i < n;i ++)
		    arr[i] = in.nextInt();
		    
		    int left[] = getLeft(arr, n);
		    int right[] = getRight(arr, n);
		    
		    int water = 0;
		    for(int i = 0; i < n ; i ++)
		    water += (Math.min(left[i], right[i]) - arr[i]);
		    
		    System.out.println(water);
		}
```
```python
# Method-2
t = int(input())
for _ in range(t):

    n = int(input())
    arr = list(map(int, input().split()))
    
    leftm = rightm = 0
    low, high = 0, n-1
    water = 0
    
    while low <= high:
        if arr[low] < arr[high]:
            if arr[low] > leftm:
                leftm = arr[low]
            else:
                water += leftm - arr[low]
            low += 1
        else:
            if arr[high] > rightm:
                rightm = arr[high]
            else:
                water += rightm - arr[high]
            high -= 1
    
    print(water)
```
## 6. Design a data-structure SpecialStack that supports all the stack operations like push(), pop(), isEmpty(), isFull() and an additional operation getMin() which should return minimum element from the SpecialStack. Your task is to complete all the functions, using stack data-Structure. Expected time and space complexity is O(1)
```python
mine = float('inf')

def push(arr, ele):
    
    global mine
    if len(arr) == 0:
        mine = ele
        arr.append(ele)
    elif mine > ele:
        arr.append(2*ele - mine)
        mine = ele
    else:
        arr.append(ele)


def pop(arr):
    global mine
    # Code here
    if len(arr) == 0:
        return -1
    if arr[-1] >= mine:
        return arr.pop()
    elif arr[-1] < mine:
        mine = 2*mine - arr[-1]
        return arr.pop()


def isFull(n, arr):
    # Code here
    global mine
    return len(arr) == n

def isEmpty(arr):
    
    return len(arr) == 0

def getMin(n, arr):
    global mine
    if len(arr) == 0:
        return -1
    return mine

```
## 7. 