## Text Alignment
thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


## Validating Postal Codes
regex_integer_in_range = r"^[1-9]\d{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=(\d)\1)"	# Do not delete 'r'.

## Matrix Script
#!/bin/python3

import math
import os
import random
import re
import sys

import re


first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

s = ''
for j in range(m):
    for i in range(n):
        s += matrix[i][j]


s = re.sub(r'(?<=\w)[\W]+?(?=\w)', ' ', s)
print(s)
## Validating Credit Card Numbers
import re

n = int(input())

for _ in range(n):
    s = input()
    

    p1 = r'^[456]\d{3}(-?)\d{4}\1\d{4}\1\d{4}$'
    m1 = re.search(p1, s)
    
    s = re.sub(r'\D', '', s)
    m2 = re.search(r'([\d])\1\1\1', s)
    
    if m1 and not m2:
        print('Valid')
    else:
        print('Invalid')

## Validating UID
import re

n = int(input())

for _ in range(n):
    s = input()
    
    p1 = r"^(.*?[A-Z]){2,}.*$" # 2 upper case
    p2 = r"^(.*?[0-9]){3,}.*$" # 3 digits
    p3 = r"(?:([a-zA-Z\d])(?!.*\1)){10}" # 10 chars and no repeating
    
    if re.search(p1, s) and re.search(p2, s) and re.search(p3, s):
        print('Valid')
    else:
        print('Invalid')

## Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser

# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        if attrs:
            for att in attrs: 
                print("->", att[0], '>', att[1])
                
    def handle_startendtag(self, tag, attrs):
        print(tag)
        if attrs:
            for att in attrs: 
                print("->", att[0], '>', att[1])

html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

## HTML Parser Part 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        data = data.split('\n')
        if len(data) == 1:
            print(">>> Single-line Comment")
            print(data[0])
        else:
            print(">>> Multi-line Comment")
            for d in data: print(d)
    
    def handle_data(self, data):
        data = data.replace('\n', '') # remove spaces
        if data:
            print(">>> Data")
            print(data)

  
  
  
  
  
  
  
  
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

## HTML Parser Part 1
from html.parser import HTMLParser

n = int(input())



# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        if attrs:
            for att in attrs: 
                print("->", att[0], '>', att[1])
                
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        if attrs:
            for att in attrs: 
                print("->", att[0], '>', att[1])

parser = MyHTMLParser()

for _ in range(n):
    s = input()
    parser.feed(s)
parser.close()

## Hex Color Code
import re

n = int(input())
insideBrackets = False

for _ in range(n):
    s = input()
    
    if '{' in s: 
        insideBrackets = True 
        continue
    
    if '}' in s: insideBrackets = False
    
    if insideBrackets:
        for x in re.findall(r'#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})', s):
            print(f'#{x}')

## Validating and Parsing Email Addresses
import re

n = int(input())

for _ in range(n):
    
    s = input()
    b = re.search(r'<[a-z][\w\.-]+@[a-z]+\.[a-z]{1,3}>', s)

    if b: print(s)

## Validating phone numbers
N = int(input())

regex_pattern = r"^(7|8|9)\d{9}$"

import re
for _ in range(N):
    b = bool(re.match(regex_pattern, input()))
    if b:
        print('YES')
    else:
        print('NO')

## Validating Roman Numerals
# I, V, X, L, C, D, M

regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

## Regex Substitution
import re

N = int(input())

for _ in range(N):
    s = ''
    s2 = input()
    
    while s != s2:
        s = s2
        s2 = re.sub(r'\s&&\s', r' and ', s)
        s2 = re.sub(r'\s\|\|\s', r' or ', s2)

    print(s2)


## Re.start() & Re. end()
import re

S = input()
k = input()

f = re.finditer(r'(?=(' + k + r'))', S)
b = re.search(k, S)

if b:
    for x in f:
        start = x.start()
        end = start + len(k) -1
        print(f'({start}, {end})')
else:
    print('(-1, -1)')

## Re findall() & Re. finditer()
import re

pattern = r'(?=([QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm][AEIOUaeiou]{2,}[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm]))'

f = re.findall(pattern, input())

if f:
    for x in f: print(x[1:(len(x)-1)])
else:
    print(-1)

## Group(), Groups() & Groupdict()
import re

m = re.search(r'([0-9a-zA-Z])\1', input())

if m: print(m.group()[0]) 
else: print(-1)

## Re.split()
regex_pattern = r"[,\.]"	# Do not delete 'r'.


## Detect Floating Point Number
import re

n = int(input())
for _ in range(n):
    
    s = input()
    
    
    pattern = r'^[+-]?\.?\d+\.?\d*$'
    
    # I had to add this s!='0' because the tester had a
    # case where 0 should be a False.
    if re.match(pattern, s) and s!='0':
        print(True)
    else:
        print(False)

## Time Delta
#!/bin/python3

import math
import os
import random
import re
import sys
from datetime import datetime

# Complete the time_delta function below.
def time_delta(t1, t2):
    t1 = datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z')
    t2 = datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')
    
    return str(abs(int((t1-t2).total_seconds())))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

## Linear Algebra
import numpy

n = int(input())
A = numpy.array([input().split() for _ in range(n)], float)
print(round(numpy.linalg.det(A), 2))

## Polynomials
import numpy



P = numpy.array(input().split(), float)
x = int(input())

print(numpy.polyval(P,x))

## Inner and Outer
import numpy

A = numpy.array(input().split(), int)
B = numpy.array(input().split(), int)

print(numpy.inner(A,B))
print(numpy.outer(A,B))

## Dot and Cross
import numpy

n = int(input())
A = numpy.array([input().split() for _ in range(n)], int)
B = numpy.array([input().split() for _ in range(n)], int)


print(numpy.dot(A,B))

## Mean, Var, and Sto [+4 more]
import numpy

n, m = map(int, input().split())
arr = numpy.array([input().split() for _ in range(n)], int)

print(numpy.mean(arr, axis=1))
print(numpy.var(arr, axis=0))
std = numpy.std(arr)
if std > 0:
    print(f'{std:.11f}')
else:
    print(std)

## Min and Max
import numpy

n, m = map(int, input().split())

arr = numpy.array([input().split() for _ in range(n)], int)
print(numpy.max(numpy.min(arr, axis=1)))

## Sum and Prod
import numpy

n, m = map(int, input().split())

arr = numpy.array([input().split() for _ in range(n)], int)
print(numpy.prod(numpy.sum(arr, axis=0)))

## Floor, Ceil and Rint [+1 more]
import numpy
numpy.set_printoptions(legacy='1.13')

arr = numpy.array(input().split(), float)

print(numpy.floor(arr))
print(numpy.ceil(arr))
print(numpy.rint(arr))

## Array Mathematics
import numpy



n, m = map(int, input().split())

A = []
B = []

for _ in range(n): A.append(list(map(int, input().split())))
for _ in range(n): B.append(list(map(int, input().split())))

A = numpy.array(A)
B = numpy.array(B)

print(A+B)
print(A-B)
print(A*B)
print(A//B)
print(A%B)
print(A**B)

## Eye and Identity
import numpy
numpy.set_printoptions(legacy='1.13')

n, m = map(int, input().split())

print(numpy.eye(n,m))

## Zeros and Ones
import numpy

args = tuple(map(int, input().split()))

print(numpy.zeros(args, dtype = int))
print(numpy.ones(args, dtype = int))

## Concatenate
import numpy

N,M,P = map(int, input().split())
arr = []
for _ in range(N+M):
    arr.append(list(map(int,input().split())))
    
print(numpy.array(arr))

## Transpose and Flatten
import numpy as np


N, M = map(int, input().split())
arr = []

for _ in range(N):
    arr.append(list(map(int, input().split())))
    
arr = np.array(arr)
print(np.transpose(arr))
print(arr.flatten())

## Shape and Reshape
import numpy

arr = numpy.array(input().strip().split(' '), int)
print(numpy.reshape(arr,(3,3)))

## Arrays
def arrays(arr):
    return numpy.flip(numpy.array(arr,float))

## Decorators 2 Name Directory
def person_lister(f):
    def inner(people):
      
        return [f(p) for p in sorted(people, key = lambda x: int(x[2]))]
        
    return inner

## Standardize Mobile Number Using
def wrapper(f):

    def fun(l):
        
        for i in range(len(l)):
            number = l[i]
            
            if not len(number) == 10:
                number = number[(len(number)-10):len(number)]
                
            l[i] = f'+91 {number[:5]} {number[5:]}'
            
        f(l)
    
    return fun

## Decorators
maxdepth = 0
def depth(elem, level):
    level += 1
    global maxdepth
    # your code goes here
    
    for child in elem:
        depth(child, level)

    if level > maxdepth: maxdepth = level


## XML2 Find the Maximum Depth
maxdepth = 0
def depth(elem, level):
    level += 1
    global maxdepth
    # your code goes here
    
    for child in elem:
        depth(child, level)

    if level > maxdepth: maxdepth = level



## XML 1 Find the Score
def get_attr_number(node):
    count = 0
    
    for n in node.iter():
        count += len(n.attrib)

    return count

## Map and Lambda Function
cube = lambda x: x**3

def fibonacci(n):
    li = [0,1]
    
    for i in range(n-2):
        li.append(li[i] + li[i+1])
        
    return li[:n]

## ginorts
s = ''.join(sorted(input()))

lowers = ''.join([c if c.islower() else '' for c in s])
uppers = ''.join([c if c.isupper() else '' for c in s])

odds = ''.join([c if c.isnumeric() and int(c) % 2 == 1 else '' for c in s])
evens = ''.join([c if c.isnumeric() and int(c) % 2 == 0 else '' for c in s])

print(lowers+uppers+odds+evens)

## Athlete Sort
#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    
    arr.sort(key = lambda x: x[k])
    
    for a in arr:
        print(*a)

## Zipped!
N, X = map(int, input().split())

l = [list(map(float, input().split())) for x in range(X)]
z = zip(*l)
[print(sum(s) / len(s)) for s in z]

## Exceptions
T = int(input())

for _ in range(T):
    try:
        a, b = map(int, input().split())
    except ValueError as e:
        print(f'Error Code: {e}')
    else:
        try:
            print(a//b)
        except ZeroDivisionError as e:
            print(f'Error Code: {e}')
    

## Calendar Module
import calendar

date = list(map(int, input().split()))

date = calendar.weekday(date[2],date[0],date[1])
print(list(calendar.day_name)[date].upper())


## Piling Up!
from collections import deque

T = int(input())
for _ in range(T):
    isStackable = True
    last_item = 2**31 + 1
    
    n = int(input())
    blocks = deque(map(int, input().split()))
    
    left = blocks.popleft()
    right = blocks.pop()
    
    
    while blocks:
        if left >= right and left <= last_item:
            last_item = left
            left = blocks.popleft()
        elif right > left and right <= last_item:
            last_item = right
            right = blocks.pop()
        else:
            isStackable = False
            break
        
    if not((left >= right and left <= last_item) or (right > left and right <= last_item)):
        isStackable = False
    
    if isStackable:
        print('Yes')
    else:
        print('No')

## Company Logo
#!/bin/python3

import math
import os
import random
import re
import sys

from collections import Counter

if __name__ == '__main__':
    s = input()
    ch_count = Counter()
    
    for ch in ''.join(sorted(s)):
        ch_count[ch] += 1
    
    ch_count = ch_count.most_common()
    for i in range(3):
        print(ch_count[i][0], ch_count[i][1])


## Word Order
from collections import Counter
n = int(input())
word_count = Counter()
for _ in range(n):
    word = input()
    word_count[word] += 1
    
print(len(word_count))

for w in word_count:
    print(word_count[w], end=' ')
    

## Collections.deque()
from collections import deque

d = deque()
N = int(input())

for _ in range(N):
    action, *args = input().split()
    args = list(map(int, args))
    
    func = getattr(d, action)
    func(*args)
    
print(*d)

## Check Strict Superset
A = set(map(int, input().split()))
n = int(input())
issuperset = True

for _ in range(n):
    B = set(map(int, input().split()))
    if len(A) <= len(B) or not A.issuperset(B):
        issuperset = False
        break
    
print(issuperset)

## Check Subset
T = int(input())

for _ in range(T):
    _ = input()
    A = set(map(int, input().split()))
    _ = input()
    B = set(map(int, input().split()))
    print(A.issubset(B))

## Set .union() Operation
_ = input()
A = set(map(int, input().split()))
_ = input()
B = set(map(int, input().split()))

print(len(A|B))

## The Captain's Room [+4 more]
K = int(input())
A = list(map(int, input().split()))

B = set(A)
C = set(A)

for a in A:
    if a in C:
        C.remove(a)
    elif a in B:
        B.remove(a)
        
print(*B)

## Set Mutations
_ = int(input())
A = set(list(map(int, input().split())))
N = int(input())


for _ in range(N):
    action, _ = input().split()
    B = set(map(int, input().split()))
    
    func = getattr(A, action)
    func(B)
    
print(sum(A))

## Set .symmetric_difference() Operation
_ = input()
A = set(map(int, input().split()))
_ = input()
B = set(map(int, input().split()))

print(len(A^B))

## Set .difference() Operation
_ = input()
A = set(map(int, input().split()))
_ = input()
B = set(map(int, input().split()))

print(len(A-B))


## Set .intersection() Operation
_ = input()
A = set(map(int, input().split()))
_ = input()
B = set(map(int, input().split()))

print(len(A&B))

## Set .discard(), remove() & .pop()
n = int(input())
my_set = set(list(map(int, input().split())))
N = int(input())


for _ in range(N):
    action, *args = input().split()
    args = list(map(int, args))
    
    func = getattr(my_set, action)
    func(*args)
    
print(sum(list(my_set)))

## Symmetric Difference
M = int(input())
a = set(map(int, input().split()))
N = int(input())
b = set(map(int, input().split()))

for x in sorted(list(a.symmetric_difference(b))):
    print(x)

## Set .add()
N = int(input())
my_set = set()

for _ in range(N):
    my_set.add(input())
    
print(len(my_set))


## No Idea!
_ = input()
arr = list(map(int, input().split()))
A = set(map(int, input().split()))
B = set(map(int, input().split()))

happiness = 0

for x in arr:
    happiness += (x in A) - (x in B)
print(happiness)

## Collections. OrderedDict()
from collections import OrderedDict

N = int(input())
ordered_dictionary = OrderedDict()

for _ in range(N):
    item_name, net_price = input().rsplit(' ', 1)
    net_price = int(net_price)
    
    if item_name in ordered_dictionary:
        ordered_dictionary[item_name] += net_price
    else:
        ordered_dictionary[item_name] = net_price
        
for key in ordered_dictionary:
    print(key, ordered_dictionary[key])


## Collections.namedtuple()
from collections import namedtuple
N = int(input())
Point = namedtuple('Point',input().split())
average = 0
for _ in range(N):
    average += int(Point(*input().split()).MARKS) / N
print(average)


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


## Introduction to Sets
def average(array):
    ave = sum(set(array)) / len(set(array))
    return ave


## Merge the Tools!
def merge_the_tools(string, k):
    
    n = len(string)
    
    for i in range(0,n,k):
        s = set([*string[i:(i+k)]])
        s = ''.join(s)
        print(s)


## The Minion Game
def minion_game(string):
    s = 0
    k = 0
    
    for i in range(len(string)):
        if string[i] in ['A','E','I','O','U']:
            k+= (len(string) - i)
        else:
            s+= (len(string) - i)
                
    if s > k:
        print(f'Stuart {s}')
    elif k > s:
        print(f'Kevin {k}')
    else:
        print('Draw')
            
## Capitalize!

# Complete the solve function below.
def solve(s):
    s = s.split(' ')
    s = [x.capitalize() for x in s]
    return ' '.join(s)


## Alphabet Rangoli
def print_rangoli(size):
    base = 96
    width = size*4 - 3
    
    for i in range(size, 0, -1):
        pattern = f'{chr(base+i)}'
        
        for j in range(i+1,size+1):
            pattern = f'{chr(base+j)}-' +pattern+ f'-{chr(base+j)}' 
             
        print(pattern.center(width, '-'))
    
    for i in range(2, size+1):
        pattern = f'{chr(base+i)}'
        
        for j in range(i+1,size+1):
            pattern = f'{chr(base+j)}-' +pattern+ f'-{chr(base+j)}' 
        
        print(pattern.center(width, '-'))
    

## String Formatting
def print_formatted(number):
    width = log_of_2(number) + 2

    for i in range(1, number+1):
        s = ''
        s += str(i).rjust(width-1,' ')
        s += oct(i).split('o')[1].rjust(width,' ')
        s += hex(i).split('x')[1].upper().rjust(width,' ')
        s += bin(i).split('b')[1].rjust(width,' ')
        print(s)
        

def log_of_2(num):
    res = num
    count = 0

    while res / 2 >= 1:
        res /= 2
        count += 1

    return count
    

## Designer Door Mat
# Enter your code here. Read input from STDIN. Print output to STDOUT
N, M = input().split()
N = int(N)
M = int(M)

motif = '.|.'
pattern = motif

p = N // 2

for _ in range(p):
    print(pattern.center(M,'-'))
    pattern = pattern + motif + motif
    
print('WELCOME'.center(M,'-'))

for _ in range(p):
    pattern = pattern[6:]
    print(pattern.center(M,'-'))

## Text Wrap
def wrap(string, max_width):
    s = textwrap.wrap(string,max_width)
    s = '\n'.join(s)
    return s


## String Validators
if __name__ == '__main__':
    s = input()
    print(any(ch.isalnum() for ch in s))
    print(any(ch.isalpha() for ch in s))
    print(any(ch.isdigit() for ch in s))
    print(any(ch.islower() for ch in s))
    print(any(ch.isupper() for ch in s))

## Find a string
def count_substring(string, sub_string):
    count = 0
    
    ## substring length
    ssl = len(sub_string)
    
    for i in range(len(string)-ssl+1):
        if string[i:(i+ssl)] == sub_string:
            count += 1
            
    return count

## Mutations
def mutate_string(string, position, character):
    s = string[:(position)] + character + string[(position+1):]
    return s


## What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#

def print_full_name(first, last):
    print(f'Hello {first} {last}! You just delved into python.')


## String Split and Join
def split_and_join(line):
    s = line.split()
    s = "-".join(s)
    
    return s
    
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

## SWAP cASE
def swap_case(s):
    
    swaped = ''
    
    for ch in s:
        if ch.islower():
            swaped = swaped + ch.upper()
        else:
            swaped = swaped + ch.lower()
            
    return swaped


## Nested Lists
if __name__ == '__main__':
    students = []

    for _ in range(int(input())):
        name = input()
        score = float(input())
        
        students.append([name, score])

    # sort according to second element
    students.sort(key = lambda x: x[1])

    lowest_grade = students[0][1]
    second_lowest_names = []

    for i in range(1,len(students)):
        if students[i][1] > lowest_grade:
            
            # put the name in the list
            second_lowest_names.append(students[i][0])
            
            # search for other students with the same grade
            for j in range(i+1, len(students)):
                if students[j][1] == students[i][1]:
                    second_lowest_names.append(students[j][0])
                else:
                    break
                
            break

    second_lowest_names.sort()
    for name in second_lowest_names:
        print(name)

## Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    
    my_tuple = ()
    
    for i in integer_list:
        my_tuple = my_tuple + (i,)
        
    print(hash(my_tuple))

## Lists
if __name__ == '__main__':
    N = int(input())
    
    my_list = []
    
    for _ in range(N):
        action, *args = input().split()
        
        if action == 'print':
            print(my_list)
        else:
            # convert args to ints
            args = list(map(int, args))
            
            # get attribute
            func = getattr(my_list, action)
            
            # pass arguments
            func(*args)

## Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
    grades = student_marks[query_name]
    avg = sum(grades) / len(grades)
    
    # print with 2 decimals
    print("{:.2f}".format(avg))

## Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
    arr = list(arr)
    arr.sort(reverse=True)
    
    maximum = arr[0]
    
    for i in range(1, len(arr)):
        if arr[i] < maximum:
            print(arr[i])
            break
    

## List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    my_list = []
    
    # The order of the loops allows to add the
    # permutations in lexicographic increasing order.
    
    for i in range(x+1):
        for j in range(y+1):
            for k in range(z+1):
                if i+j+k != n:
                    my_list.append([i,j,k])
                    
    print(my_list)



## Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    print(a+b)
    print(a-b)
    print(a*b)

## Print Function
if __name__ == '__main__':
    n = int(input())
    
    for i in range(1, n+1):
        print(i, end='')

## Write a function
def is_leap(year):
    leap = False
    
    # Write your logic here
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                leap = True
        
        else:
            leap = True
    
    return leap


## Loops
if __name__ == '__main__':
    n = int(input())
    
    for i in range(n):
        print(i**2)

## Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    print(a//b)
    print(a/b)

## Python If-Else
#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 == 1:
        print('Weird')
    elif n in list(range(6,21)):
        print('Weird')
    else:
        print('Not Weird')


## Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")

