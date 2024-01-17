# line_count.py
import sys

count = 0
for line in sys.stdin:
   count += 1

# a impressão para sys.stdout
print(count)