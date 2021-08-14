from multiprocessing import Pool
import time

def f(x):
    time.sleep(2)
    return x*x

a = [1,2,3,4,5,6,7]
b = []
if __name__ == '__main__':
    with Pool(10) as p:
        t = time.time()
        b = p.map(f, a)
        print(f"delta = {time.time()-t}")
    print(b)