import numpy as np

y = -1
print("\nwrite n:")
n = int(input())
if n <= 0:
    print("\nERROR")
    exit()

print("\nwrite i:")
i = int(input())

print("\nwrite A:")
A = []
for row in range(n):
    print(f"\nStroka {row + 1}:")
    A.append(list(map(float, input().split())))
A = np.array(A)

if A.shape != (n, n):
    print("\nERROR")
    exit()

A1 = np.linalg.inv(A)

print("\nwrite x:")
x = np.array(list(map(float, input().split()))).reshape(n, 1)

A_modified = A.copy()
A_modified[:, i-1] = x.flatten()

print("\nA:")
print(A)
print("\nA_modified:")
print(A_modified)

l = np.dot(A1, x)

if l[i-1, 0] != 0:
  l1= l.copy()
  l1[i-1:] = y
  l2 = -1/l[i-1] * l1
  Q = np.eye(n)
  Q_modified = Q.copy()
  Q_modified[:, i-1] = l2.flatten()
  A_modified1 =  np.dot(Q_modified,A1)
  print("\nA_modified1:")
  print(A_modified1)
else:
  print ("\nA_modified1 is not exist")