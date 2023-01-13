# %%
import math
import concurrent.futures

import multiprocessing as mp

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    
    def _get_gen(n):
        for _ in range(n):
            yield 112272535095293

    with concurrent.futures.ProcessPoolExecutor(mp.cpu_count()) as executor:
        for i, (prime) in enumerate(executor.map(is_prime, _get_gen(1000))):
            print(f'\r{i} {prime}', end='')
            # print(f'{int(number)} is prime: {prime}, {i}')

if __name__ == '__main__':
    main()