#!/usr/bin/env python3
def old_mirror(idx, sup):
    v = abs(idx)
    return v if v < sup else sup - (v - sup + 1)

def new_mirror(idx, sup):
    if idx < 0: return -idx
    if idx >= sup: return 2 * (sup - 1) - idx
    return idx

for sup in [5, 324, 576]:
    for idx in range(-10, sup + 10):
        o = old_mirror(idx, sup)
        n = new_mirror(idx, sup)
        if o != n:
            print("DIFF: mirror(%d, %d) = old:%d, new:%d" % (idx, sup, o, n))

print("done")
