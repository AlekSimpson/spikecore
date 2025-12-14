

def square_torus(k: int) -> dict[int, list[int]]:
    return {
        i: [
            (i//k)*k + (i%k + 1) % k,     # right
            (i//k)*k + (i%k - 1) % k,     # left
            ((i//k + 1) % k)*k + (i%k),   # down
            ((i//k - 1) % k)*k + (i%k),   # up
        ]
        for i in range(k*k)
    }




