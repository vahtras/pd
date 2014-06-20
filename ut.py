def upper_triangular(n):
    for i in range(3):
        for j in range(i, 3):
            yield i, j
