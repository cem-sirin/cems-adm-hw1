## Birthday Cake Candles
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    return candles.count(max(candles))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

    ## Number Line Jumps
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    if v1 > v2 and x2 > x1:
        while x1 < x2:
            x1 += v1
            x2 += v2
        
        if x1 == x2:
            return 'YES'
        
    elif v2 > v1 and x1 > x2:
        while x2 < x1:
            x1 += v1
            x2 += v2
        
        if x1 == x2:
            return 'YES'
    
    elif x1==x2:
        return 'YES'
    
    return 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

## Viral Advertising
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    shared = 5
    liked = shared // 2
    cum = liked
    
    for _ in range(n-1):
        shared = liked * 3
        liked = shared // 2
        cum += liked
        
    return cum

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

## Recursive Digit Sum
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    n = str(n)
    n = [int(x) for x in n]
    n = sum(n)
    n *= k
    
    while n > 9:
        n = str(n)
        n = [int(x) for x in n]
        n = sum(n)
    
    return n

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

## Insertion Sort Part
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    unsorted = arr[n-1]
    
    idx = n-2
    while arr[idx] > unsorted and idx >= 0:
        arr[idx+1] = arr[idx]
        print(*arr)
        idx -= 1
    
    arr[idx+1] = unsorted
    print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

## DefaultDict Tutorial
from collections import defaultdict

n, m = map(int, input().split())

A = defaultdict(list)
B = []

for i in range(1, n+1):
    A[input()].append(str(i))

for i in range(1, m+1):
    B.append(input())


for ch in B:
    if A[ch]:
        print(' '.join(A[ch]))
    else:
        print(-1)

## collections. Counter()
from collections import Counter

# number of shoes
X = int(input())
# counter of shoe sizes
shoe_sizes = Counter(map(int, input().split()))
# number of customers
N = int(input())

money = 0

for _ in range(N):
    info = list(map(int, input().split()))
    want = int(info[0])
    price = int(info[1])
    
    if shoe_sizes[want] > 0:
        money += price
        shoe_sizes[want] -= 1
        
print(money)

## Insertion Sort Part 2
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    
    for i in range(1,n):
    
        for j in range(i,0,-1):
            if arr[j] < arr[j-1]:
                temp = arr[j-1]
                arr[j-1] = arr[j]
                arr[j] = temp
            else:
                break
        
        print(*arr)
                

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)