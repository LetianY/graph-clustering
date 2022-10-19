from itertools import combinations
from random import randint

n = 64

for u, v in combinations(range(n), 2):
    if randint(0, 32) == 0:
        print(f"{u} {v}")