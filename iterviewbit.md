# InterviewBit Questions Continued

## 1. You are given an array (zero indexed) of N non-negative integers, A0, A1 ,…, AN-1.
Find the minimum sub array Al, Al+1 ,…, Ar so if we sort(in ascending order) that sub array, then the whole array should get sorted.
If A is already sorted, output -1.

Example :

Input 1:

A = [1, 3, 2, 4, 5]

Return: [1, 2]

Input 2:

A = [1, 2, 3, 4, 5]

Return: [-1]

```python
#Method 1
def subUnsort(self, A):
        si = -1
        ei = 0
        max1 = 0
        min1 = max(A)
        minind = -1
        for i in range(1,len(A)):
            if A[i] < A[i - 1] or A[i] < max1:
                if si == -1:
                    si = i - 1
                ei = i
                min1 = min(min1,A[i])
            max1 = max(max1,A[i])
            
        if si == -1:
            return [-1]
        else:
            for i in range(0,si):
                if A[i] > min1:
                    minind = i
                    break
            if minind < si and minind != -1:
                si = minind
            return [si,ei]
#Method 2
def subUnsort(self, a):
        n = len(a)
        #sort the array inplace
        sort = sorted(a)
        #result
        res = []
        #iterate through the input
        for i in range(n):
            if a[i] != sort[i]:
                res.append(i)
        if len(res) == 0:
            res.append(-1)
            return res
        else:
            return [res[0], res[-1]]
```

## 2. Given an unsorted integer array, find the first missing positive integer. Example: Given [1,2,0]return 3, [3,4,-1,1] return 2, [-8, -7, -6] returns 1. Your algorithm should run in O(n) time and use constant space.
```python
def firstMissingPositive(self, a):
        a = list(filter(lambda x: x > 0, a))
        if len(a) == 0:
            return 1
        for i in range(len(a)):
            if abs(a[i]) - 1 < len(a) and a[abs(a[i])-1] > 0:
                a[abs(a[i]) - 1] = - a[abs(a[i]) - 1]
        
        for i in range(len(a)):
            if a[i] > 0:
                return i+1
        
        return len(a)+1
```
## 3. Suppose a sorted array A is rotated at some pivot unknown to you beforehand. (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2). Find the minimum element. The array will not contain duplicates.
```python
def find_min(self, a):
        n = len(a)
        low, high = 0, n-1
        
        while(low <= high):
            mid = (low+high)//2
            ne = (mid+1)%n
            pe = (mid + n -1)%n
            if a[low] <= a[high]:
                return low
            elif a[ne] >= a[mid] and a[pe] >= a[mid]:
                return mid
            elif a[mid] <= a[high]:
                high = mid-1
            else:
                low = mid + 1
        return -1
    def findMin(self, a):
        return a[self.find_min(a)]

```

## 4. Given a matrix of integers A of size N x M in which each row is sorted. Find an return the overall median of the matrix A. Note: No extra memory is allowed. Note: Rows are numbered from top to bottom and columns are numbered from left to right.

```python
def findMedian(self, a):
        maxe = 0
        mine = a[0][0]
        r = len(a)
        c = len(a[0])
        for i in range(r):
            if a[i][0] < mine:
                mine = a[i][0]
            if a[i][c-1] > maxe:
                maxe = a[i][c-1]
                
        desired = (r*c + 1)//2
        
        while mine < maxe:
            mid = (mine + maxe)//2
            place = 0
            for i in range(r):
                place += bisect_right(a[i], mid)
            if place < desired:
                mine = mid+1
            else:
                maxe = mid
        return mine
```

## 5. (Painter's Partition Problem) Given 2 integers A and B and an array of integars C of size N. Element C[i] represents length of ith board. You have to paint all N boards [C0, C1, C2, C3 … CN-1]. There are A painters available and each of them takes B units of time to paint 1 unit of board. Calculate and return minimum time required to paint all boards under the constraints that 
1. any painter will only paint contiguous sections of board.
2. painters cannot share a board to paint. That is to say, a board cannot be painted partially by one painter, and partially by another. A painter will only paint contiguous boards. Which means a configuration where painter 1 paints board 1 and 3 but not 2 is invalid. 

Return the ans % 10000003

```python
def isok(A,C,val):
    i=0
    while i < len(C):
        x=val
        while i < len(C) and x>=C[i] :
            x-=C[i]
            i+=1
        A-=1
        if A<0:
            return False
    return True
def paint(self, A, B, C):
        l,r=max(C),sum(C)
        while l<r:
            mid=(l+r)//2
            if isok(A,C,mid):
                r=mid
            else:
                l=mid+1
        return (l*B)%10000003
```

## 6. Given an integar A. Compute and return the square root of A. If A is not a perfect square, return floor(sqrt(A)). DO NOT USE SQRT FUNCTION FROM STANDARD LIBRARY
```python
def sqrt(self, A):
        if A == 0:
            return 0
        if A >= 1 and A <= 3:
            return 1
        low = 1
        high = A//2
        
        while low <= high:
            mid = low + (high-low)//2
            if (mid*mid == A):
                return mid
            if (mid*mid < A):
                low = mid+1
                val = mid
            else:
                high = mid-1
        
        return val
```

## 7. The count-and-say sequence is the sequence of integers beginning as follows: 1, 11, 21, 1211, 111221, ... 1 is read off as one 1 or 11. 11 is read off as two 1s or 21. 21 is read off as one 2, then one 1 or 1211. Given an integer n, generate the nth sequence. Note: The sequence of integers will be represented as a string. Example: if n = 2, the sequence is 11.
```python
def countAndSay(self, n):
        if n == 0:
            return ''
        res = '1'
        while n > 1:
            curr = ''
            i = 0
            while i < len(res):
                count = 1
                while (i+1 < len(res)) and res[i] == res[i+1]:
                    i+=1
                    count += 1
                curr += str(count)+res[i]
                i += 1
            res = curr
            n -= 1
        return res
```

## 8. Justify Text. Given an array of words and a length L, format the text such that each line has exactly L characters and is fully (left and right) justified. You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ‘ ‘ when necessary so that each line has exactly L characters. Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right. For the last line of text, it should be left justified and no extra space is inserted between words. Your program should return a list of strings, where each string represents a single line. Example: words: ["This", "is", "an", "example", "of", "text", "justification."] and L: 16. Return the formatted lines as: ["This    is    an", "example  of text","justification.  "].
```python
def fullJustify(self, a, b):
        if len(a) == 0:
            return ""
        final_ans = []
        string = ""
        j = 0
        a.append("")
        for i in a:
            if i == "" or len(i)+len(string) >= b:
                string = string.strip()
                length_of_string = len(string)-j+1
                number_of_spaces = j-1
                extra_spaces = b - length_of_string
                string = string.split()
                if number_of_spaces != 0:
                    spaces = extra_spaces//number_of_spaces
                    remainder_space = extra_spaces%number_of_spaces
                    k = 0
                    while remainder_space > 0:
                        string[k] = string[k] + ' '
                        remainder_space -= 1
                        k += 1
                    string = (" "*spaces).join(string)
                else:
                    spaces = extra_spaces
                    string = " ".join(string)
                    string += " "*spaces
                    remainder_space = 0
                final_ans.append(string)
                string = i
                j = 1
            else:
                string += " "+i
                string = string.strip()
                j += 1
                
        final_ans[-1] = " ".join(final_ans[-1].split())
        final_ans[-1]+= " "*(b - len(final_ans[-1]))
        if final_ans[0] == "":
            final_ans = final_ans[1:]
        return final_ans
```
## 9. Given an integer A, convert it to a roman numeral, and return a string corresponding to its roman numeral version. 
```python
def intToRoman(self, a):
        sym =['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
        arr =[1000,900,500,400,100,90,50,40,10,9,5,4,1]
        res=""
        while(a>0):
            i = 0
            while i < 13:
                if(a>=arr[i]):
                    res+=sym[i]
                    a=a-arr[i]
                    i = i-1
                i+=1
        return res
```

## 10. Implement atoi to convert a string to an integer. Example : Input : "9 2704" Output : 9. Note: string contain whitespace characters before the number 2. the string can have garbage characters after the number. Ignore it. 3. If no numeric character is found before encountering garbage characters return 0. 4. If the integer overflows, Return INT_MAX if the number is positive, INT_MIN otherwise.
```python
def atoi(self, A):
        sign = 1
        j = 0
        
        # Process potential spaces
        # Note: using the string "strip" method makes a copy
        # which we prefer to avoid to save memory
        while j < len(A) and A[j] == " ":
            j += 1
            
        # If there are only spaces, return 0    
        if j == len(A):
            return 0
        
        # Process potential +/- sign
        if A[j] == "+":
            j += 1
        elif A[j] == '-':
            j += 1
            sign = -1
        
        # Process digits
        start = j
        while j < len(A) and A[j].isnumeric():
            j += 1
        
        # If there are no digits, return 0
        if start == j:
            return 0
        
        r = sign * int(A[start:j])
        return max(-2147483648, min(r, 2147483647))
```
## 11. Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target. Return the sum of the three integers. Assume that there will only be one solution Example: given array S = {-1 2 1 -4}, and target = 1. The sum that is closest to the target is 2. (-1 + 2 + 1 = 2)
```python
def threeSumClosest(self, a, b):
        n = len(a)
        bestsum = 1000000000
        sum_ = 0
        a.sort()
        for i in range(n-2):
            start = i+1
            end = n-1
            while start<end:
                sum_ = a[i]+a[start]+a[end]
                if abs(b - sum_) < abs(b - bestsum):
                    bestsum = sum_
                elif sum_>b:
                    end -= 1
                else:
                    start += 1
                    
        return bestsum
```

## 12. Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). 'n' vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water. Your program should return an integer which corresponds to the maximum area of water that can be contained ( Yes, we know maximum area instead of maximum volume sounds weird. But this is 2D plane we are working with for simplicity ). Input : [1, 5, 4, 3] Output : 6 Explanation : 5 and 3 are distance 2 apart. So size of the base = 2. Height of container = min(5, 3) = 3. So total area = 3 * 2 = 6.
```python
def maxArea(self, a):
        n = len(a)
        i, j = 0, n-1
        max_area = j*min(a[i], a[j])
        while i < j:
            if (j-i)*min(a[i], a[j]) > max_area:
                max_area = (j-i)*min(a[i], a[j])
            if a[i] < a[j]:
                i += 1
            else:
                j -= 1
        return max_area
```

## 13. Given a singly linked list, determine if its a palindrome. Return 1 or 0 denoting if its a palindrome or not, respectively. Space complexity should be constant.
```python
def lPalin(self, A):
        count = 0
        node = A
        while node:
            count += 1
            node = node.next
        prev = None
        curr = A
        for i in range(count // 2):
            tmp = curr.next
            curr.next = prev
            prev, curr = curr, tmp
        if count % 2 == 1:
            curr = curr.next
        while curr:
            if curr.val != prev.val:
                return 0
            curr = curr.next
            prev = prev.next
        return 1
```
## 14. Given a sorted linked list, delete all duplicates such that each element appear only once.
```python
def deleteDuplicates(self, head):
        curr = head
        while curr:
            while curr.next and curr.val == curr.next.val:
                curr.next = curr.next.next
            curr = curr.next
            
        return head
```

## 15. Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list. For example, Given 1->2->3->3->4->4->5, return 1->2->5. Given 1->1->1->2->3, return 2->3.
```python
def deleteDuplicates(self, A):
        curr = A
        head = prev = ListNode(None)
        head.next = curr
        while curr and curr.next:
            if curr.val == curr.next.val:
                while curr and curr.next and curr.val == curr.next.val:
                    curr = curr.next
                # still one one of duplicate values left so advance
                curr = curr.next
                prev.next = curr
            else:
                prev = prev.next
                curr = curr.next
        return head.next
```
## 16. Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists, and should also be sorted.
```python
def mergeTwoLists(self, A, B):
        start = ListNode(None)
        end = start
        
        while(A and B):
            if(A.val < B.val):
                end.next = A
                A = A.next
            else:
                end.next = B
                B = B.next
            end = end.next        
        if(A):
            end.next = A
        elif(B):
            end.next = B
        
        return start.next
```
## 17. Given a linked list, remove the nth node from the end of list and return its head. For example, Given linked list: 1->2->3->4->5, and n = 2. After removing the second node from the end, the linked list becomes 1->2->3->5. Note: If n is greater than the size of the list, remove the first node of the list.
```python
def removeNthFromEnd(self, head, x):
        slow = head
        fast = head
        
        count = 0
        while count < x and fast:
            fast = fast.next
            count += 1
        
        if fast == None:
            if head != None:
                return head.next
            return None
        
        prev = slow
        while fast!= None:
            fast = fast.next
            prev = slow
            slow = slow.next
            
        prev.next = slow.next
        
        return head
```
## 18. Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x. You should preserve the original relative order of the nodes in each of the two partitions. For example, Given 1->4->3->2->5->2 and x = 3, return 1->2->2->4->3->5.
```python
def partition(self, head, x):
        
        curr = head
        start1 = start2 = tail1 = tail2 = None
        while curr != None:
            if curr.val < x:
                if start1 == None:
                    start1 = tail1 = ListNode(curr.val)
                else:
                    tail1.next = ListNode(curr.val)
                    tail1 = tail1.next
            else:
                if start2 == None:
                    start2 = tail2 = ListNode(curr.val)
                else:
                    tail2.next = ListNode(curr.val)
                    tail2 = tail2.next
            curr = curr.next
        if tail1 == None:
            return start2
        tail1.next = start2
        return start1
```

## 19. Perform Insertion Sort on LinkedList.
```python
def insert(head, n):
    t = head
    prev = None
    while t != None:
        if t.val > n.val:
            if prev != None:
                prev.next = n
            n.next = t
            return
        prev = t
        t = t.next
    prev.next = n
    n.next = None
    return
def insertionSortList(self, head):
        curr = head
        prev = None
        next = None
        new_list = ListNode(float('-inf'))
        prev = new_list
        
        while curr != None:
            next = curr.next
            insert(new_list, curr)
            curr = next
            
        return new_list.next
```

## 20. Merge Sort on LinkedList.
```python
def merge(l1, l2):
    
    if l1 == None:
        return l2
    if l2 == None:
        return l1
    final_list = None
    prev = None
    
    while l1 != None and l2 != None:
        if l1.val < l2.val:
            if final_list == None:
                final_list = l1
                prev = l1
            else:
                prev.next = l1
                prev = l1
            l1 = l1.next
        else:
            if final_list == None:
                final_list = l2
                prev = l2
            else:
                prev.next = l2
                prev = l2
            l2 = l2.next
    
    if l2 != None:
        prev.next = l2
    elif l1 != None:
        prev.next = l1
    return final_list
def sortList(self, head):
        if head == None or head.next == None:
            return head
        
        temp = slow = fast = head
        
        while fast != None and fast.next != None:
            temp = slow
            slow = slow.next
            fast = fast.next.next
            
        temp.next = None
        
        l1 = self.sortList(head)
        l2 = self.sortList(slow)
        
        return merge(l1, l2)
```

## 21. Given a linked list, return the node where the cycle begins. If there is no cycle, return null. Try solving it using constant additional space.
```python
def detectCycle(self, head):
        slow = fast = head
        while slow and fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        if slow != fast:
            return None
            
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
            
        return slow
```
## 22. You are given two linked lists representing two non-negative numbers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list. Input: (2 -> 4 -> 3) + (5 -> 6 -> 4) Output: 7 -> 0 -> 8 . i.e. 342 + 465 = 807. Make sure there are no trailing zeros in the output list. So, 7 -> 0 -> 8 -> 0 is not a valid response even though the value is still 807.
```python
def addTwoNumbers(self, num1, num2):
        carry = 0
        start1 = num1
        start2 = num2
        
        result = None
        res_tail = result
        
        while start1 and start2:
            add = start1.val + start2.val + carry
            carry = add//10
            num = add%10
            
            node = ListNode(num)
            
            if result == None:
                result = res_tail = node
            else:
                res_tail.next = node
                res_tail = node
                
            start1 = start1.next
            start2 = start2.next
                
        while start1:
            add = start1.val + carry
            carry = add//10
            num = add%10
            # if num == 0:
            #     break
            node = ListNode(num)
            if result == None:
                result = res_tail = node
            else:
                res_tail.next = node
                res_tail = node
                
            start1 = start1.next
        
        while start2:
            add = start2.val + carry
            carry = add//10
            num = add%10
            # if num == 0:
            #     break
            node = ListNode(num)
            if result == None:
                result = res_tail = node
            else:
                res_tail.next = node
                res_tail = node
                
            start2 = start2.next
            
        if carry != 0:
            node = ListNode(carry)
            res_tail.next = node
            res_tail = node
                
        return result
```
## 23. Reverse a linked list from position m to n. Do it in-place and in one-pass. For example: Given 1->2->3->4->5->NULL, m = 2 and n = 4, return 1->4->3->2->5->NULL. Note: Given m, n satisfy the following condition: 1 ≤ m ≤ n ≤ length of list. Note 2: Usually the version often seen in the interviews is reversing the whole linked list which is obviously an easier version of this question.
```python
def reverseBetween(self, A, B, C):
        head = A
        current = A
        prv = None
        nxt = None
        
        step = 1
        
        while current is not None:
            
            if step < B:
                prv = current
                current = current.next

            if step >= B and step <= C:
                if step == B:
                # this is a start of the reversed list
                    last_non_reversed = prv
                    last_reversed = current
                
                if step == C:
                    # this is the end of the reversed list
                    first_reversed = current
                    first_non_reversed = current.next
                
                # part that does reverse
                nxt = current.next
                current.next = prv
                prv = current
                current = nxt

            if step > C:
                # We can skip these steps
                break
            
            step += 1
        
        if last_non_reversed is not None:
            last_non_reversed.next = first_reversed
        else:
            head = prv
            
        last_reversed.next = first_non_reversed
            
        return head
```

## 24. Given a singly linked list. L: L0 → L1 → … → Ln-1 → Ln, reorder it to: L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → … You must do this in-place without altering the nodes’ values. For example, Given {1,2,3,4}, reorder it to {1,4,2,3}.
```python
def reverse(l):
    curr = l
    prev = next = None
    
    while curr != None:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next
        
    return prev
def reorderList(self, head):
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        second_list = slow.next
        slow.next = None
        
        first_list = head
        second_list = reverse(second_list)
        
        fl = first_list
        sl = second_list
        
        fn = sn = None
        
        while fl and sl:
            fn = fl.next
            sn = sl.next
            fl.next = sl
            sl.next = fn
            fl = fn
            sl = sn
            
        return head
```
## 25. Given a singly linked list and an integer K, reverses the nodes of the list K at a time and returns modified linked list. NOTE : The length of the list is divisible by K.
```python
def reverseList(self, head, k):
        curr = head
        next = prev = None
        count = 0
        
        while curr and count < k:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
            count += 1
            
        if next:
            head.next = self.reverseList(next, k)
            
        return prev
```

## 26. Given a linked list, swap every two adjacent nodes and return its head. For example, Given 1->2->3->4, you should return the list as 2->1->4->3.
```python
def swapPairs(self, head):
        res = head
        curr = head
        prev = next = None
        while curr and curr.next:
            t = curr
            n = curr.next
            t.next = n.next
            n.next = t
            
            if prev:
                prev.next = n
            else:
                res = n
                
            prev = curr
            curr = curr.next
            
        return res
```

## 27. Given a list, rotate the list to the right by k places, where k is non-negative. For example: Given 1->2->3->4->5->NULL and k = 2, return 4->5->1->2->3->NULL.
```python
# A: head of linkedlist
# B: no. of times to perform rotation
def rotateRight(self, A, B):
        head = A
        last = None
        length = 0
        
        while A:
            last = A
            A = A.next
            length += 1
            
        B %= length
        
        if B == 0:
            return head
        
        cur = head
        
        # get to the point where you will detach the list and rotate it
        for i in range(length - B - 1):
            cur = cur.next
        
        rotated_head = cur.next
        cur.next = None
        last.next = head
        
        return rotated_head
```

## 28. Reverse Alternate K Nodes in a LinkedList.
```python
def kAltReverse(head, k) :  
    current = head  
    next = None
    prev = None
    count = 0
  
    #1) reverse first k nodes of the linked list  
    while (current != None and count < k) :  
        next = current.next
        current.next = prev  
        prev = current  
        current = next
        count = count + 1; 
      
    # 2) Now head pos to the kth node.  
    # So change next of head to (k+1)th node 
    if(head != None):  
        head.next = current  
  
    # 3) We do not want to reverse next k  
    # nodes. So move the current  
    # poer to skip next k nodes  
    count = 0
    while(count < k - 1 and current != None ):  
        current = current.next
        count = count + 1
      
    # 4) Recursively call for the list  
    # starting from current.next. And make 
    # rest of the list as next of first node  
    if(current != None):  
        current.next = kAltReverse(current.next, k)  
  
    # 5) prev is new head of the input list  
    return prev
```

## 29. Given a linked list A , reverse the order of all nodes at even positions. Example: Input: 1->2->3->4->5->6->7. Output:  1 -> 6 -> 3 -> 4 -> 5 -> 2 -> 7 .
```python
def reverse(root):
    curr = root
    prev = next = None
    
    while curr:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next
        
    return prev
def solve(self, head):
        
        odd = head
        even = head.next
        temp1 = odd
        temp2 = even
        
        while temp1 and temp1.next and temp2 and temp2.next:
            temp1.next = temp1.next.next
            temp1 = temp1.next
            temp2.next = temp2.next.next
            temp2 = temp2.next
            
        if temp1:
            temp1.next = None
        if temp2:
            temp2.next = None
            
        even = reverse(even)
        
        ans = odd
        
        while even:
            x = odd.next
            y = even.next
            odd.next = even
            even.next = x
            odd = x
            even = y
            
        return ans
```
## 30. You are given an array A containing N integers. The special product of each ith integer in this array is defined as the product of the following:

    LeftSpecialValue: For an index i, it is defined as the index j such that A[j]>A[i] and (i>j). If multiple A[j]'s are present in multiple positions, the LeftSpecialValue is the maximum value of j.
    RightSpecialValue: For an index i, it is defined as the index j such that A[j]>A[i] and (j>i). If multiple A[j]'s are present in multiple positions, the RightSpecialValue is the minimum value of j.

Write a program to find the maximum special product of any integer in the array.

NOTE: As the answer can be large, output your answer modulo 109 + 7.

```python
# Used the concept of nearest greater to left and nearest greater to right
def maxSpecialProduct(self, arr):
        n = len(arr)
        
        ngr = [] #nearest greater to right
        i = n-1
        ngr_stk = []
        
        while i >= 0:
            if len(ngr_stk) == 0:
                ngr.append(-1)
            elif arr[ngr_stk[-1]] > arr[i]:
                ngr.append(ngr_stk[-1])
            else:
                while len(ngr_stk) != 0 and arr[ngr_stk[-1]] <= arr[i]:
                    del ngr_stk[-1]
                if len(ngr_stk) == 0:
                    ngr.append(-1)
                else:
                    ngr.append(ngr_stk[-1])
            ngr_stk.append(i)
            i -= 1
            
        ngr = ngr[::-1]
        
        ngleft = [] # nearest greater to left
        ngleft_stk = []
        i = 0
        
        while i < n:
            if len(ngleft_stk) == 0:
                ngleft.append(-1)
            elif arr[ngleft_stk[-1]] > arr[i]:
                ngleft.append(ngleft_stk[-1])
            else:
                while len(ngleft_stk) != 0 and arr[ngleft_stk[-1]] <= arr[i]:
                    del ngleft_stk[-1]
                if len(ngleft_stk) == 0:
                    ngleft.append(-1)
                else:
                    ngleft.append(ngleft_stk[-1])
            ngleft_stk.append(i)
            i += 1
            
        ans = 0
        for i in range(n):
            if ngr[i] < 0 and ngleft[i] < 0:
                continue
            ans = max(ans, ngr[i]*ngleft[i])
        
        return ans%1000000007 if ans >= 0 else 0
```

## 31. Given an array, find the nearest smaller element G[i] for every element A[i] in the array such that the element has an index smaller than i.

More formally,

    G[i] for an element A[i] = an element A[j] such that 
    j is maximum possible AND 
    j < i AND
    A[j] < A[i]

```python
def prevSmaller(self, arr):
        n = len(arr)
        
        stk = []
        
        out = []
        
        for i in range(n):
            if len(stk) == 0:
                out.append(-1)
            elif stk[-1] < arr[i]:
                out.append(stk[-1])
            else:
                while len(stk) != 0 and stk[-1] >= arr[i]:
                    del stk[-1]
                if len(stk) == 0:
                    out.append(-1)
                else:
                    out.append(stk[-1])
                    
            stk.append(arr[i])
            
        return out
```
## 32. Maximum Area of a histogram.
```python
def largestRectangleArea(self, arr):
        n = len(arr)
        
        stk = []
        
        nsl = []
        
        for i in range(n):
            if len(stk) == 0:
                nsl.append(-1)
            elif arr[stk[-1]] < arr[i]:
                nsl.append(stk[-1])
            else:
                while len(stk) != 0 and arr[stk[-1]] >= arr[i]:
                    del stk[-1]
                if len(stk) == 0:
                    nsl.append(-1)
                else:
                    nsl.append(stk[-1])
                    
            stk.append(i)
            
            
        stk = []
        
        nsr = []
        
        for i in range(n-1,-1,-1):
            if len(stk) == 0:
                nsr.append(n)
            elif arr[stk[-1]] < arr[i]:
                nsr.append(stk[-1])
            else:
                while len(stk) != 0 and arr[stk[-1]] >= arr[i]:
                    del stk[-1]
                if len(stk) == 0:
                    nsr.append(n)
                else:
                    nsr.append(stk[-1])
                    
            stk.append(i)
            
        nsr = nsr[::-1]
        
        width = [0]*n
        
        for i in range(n):
            width[i] = nsr[i] - nsl[i] - 1
            
        ans = -1
        for i in range(n):
            ans = max(ans, arr[i]*width[i])
        
        return ans 
```

## 33. Given a string A representing an absolute path for a file (Unix-style). Return the string A after simplifying the absolute path. Note: Absolute path always begin with ’/’ ( root directory ) and Path will not have whitespace characters. Input: A = "/a/./b/../../c/" Output: "/c".
```python
def simplifyPath(self, s):
        w = s.strip().split('/')
        stk = []
        for ch in w:
            if ch != '' and ch != '.':
                if ch == '..':
                    if len(stk) != 0:
                        del stk[-1]
                else:
                    stk.append(ch)
                    
        res = '/'
        for i in stk:
            res += i+'/'
            
        if len(res) == 1:
            return res
        else:
            return res[:-1]
```
## 34. Given a string A denoting an expression. It contains the following operators ’+’, ‘-‘, ‘\*’, ‘/’.Chech whether A has redundant braces or not. Return 1 if A has redundant braces, else return 0. Note: A will be always a valid expression.Input 1: A = "((a + b))" Output 1: 1. Input 2: A = "(a + (a + b))" Output 2: 0.
```python
def braces(self, s):
        stk = []
        for ch in s:
            if ch == '(':
                stk.append(ch)
            elif ch != ')':
                stk.append('$')
            else:
                if stk[-1] == '(':
                    return 1
                count = 0
                while len(stk) != 0 and stk[-1] != '(':
                    count += 1
                    del stk[-1]
                if len(stk) != 0:
                    del stk[-1]
                if count == 1:
                    return 1
                    
        return 0
```
## 35. Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

        push(x) – Push element x onto stack.
        pop() – Removes the element on top of the stack.
        top() – Get the top element.
        getMin() – Retrieve the minimum element in the stack.

Note that all the operations have to be constant time operations.
```python
class MinStack:
    # @param x, an integer
    def __init__(self):
        self.stk = []
        self.mine = []
        
    def push(self, x):
        self.stk.append(x)
        if len(self.mine) == 0 or self.mine[-1] >= x:
            self.mine.append(x)
            
    # @return nothing
    def pop(self):
        if len(self.mine) == 0:
            return
        x = self.stk[-1]
        self.stk.pop()
        if x == self.mine[-1]:
            self.mine.pop()

    # @return an integer
    def top(self):
        if len(self.stk) > 0:
            return self.stk[-1]
        return -1

    # @return an integer
    def getMin(self):
        if len(self.mine) > 0:
            return self.mine[-1]
        return -1
```
## 36. Given a string A denoting a stream of lowercase alphabets. You have to make new string B. B is formed such that we have to find first non-repeating character each time a character is inserted to the stream and append it at the end to B. If no non-repeating character is found then append '#' at the end of B. A = "abadbc" Output = "aabbdd". Explaination: 
	"a"      -   first non repeating character 'a'
    "ab"     -   first non repeating character 'a'
    "aba"    -   first non repeating character 'b'
    "abad"   -   first non repeating character 'b'
    "abadb"  -   first non repeating character 'd'
    "abadbc" -   first non repeating character 'd'

```python
def solve(self, s):
        que = []
        d = dict()
        
        ans = ''
        
        for ch in s:
            if ch not in d:
                que.append(ch)
                d[ch] = 1
            else:
                try:
                    que.remove(ch)
                except:
                    pass
            
            if len(que) > 0:    
                ans += que[0]
            else:
                ans += '#'
                
        return ans
```
## 37. Given an array of integers A. There is a sliding window of size B which
is moving from the very left of the array to the very right.
You can only see the w numbers in the window. Each time the sliding window moves
rightwards by one position. You have to find the maximum for each window.
```python
def slidingMaximum(self, A, B):
        A=list(A)
        if len(A)==0:
            return
        if len(A)<=B:
            return [max(A)]
        
        maxx=max(A[0:B])
        
        l=[]
        l.append(maxx)
        
        i=0
        
        while i<len(A)-B:
            if A[i+B]>=maxx:
                maxx=A[i+B]
            if A[i]==maxx and A[i+B]!=maxx:
                maxx=max(A[i+1:i+B+1])
            l.append(maxx)
            i+=1
        return l
```

## 38. 