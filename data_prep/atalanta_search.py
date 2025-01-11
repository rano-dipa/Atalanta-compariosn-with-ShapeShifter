import math

PROBS = 16
PROB_BITS = 10
DEPTH_MAX = 2

class Pte:
    def __init__(self, vmin=0, off=0, abits=0, obits=0, vcnt=0):
        self.vmin = vmin
        self.off = off
        self.abits = abits
        self.obits = obits
        self.vcnt = vcnt

pcnt = PROBS
vmax = 0
pnew = [Pte() for _ in range(PROBS + 1)]
pbest = [Pte() for _ in range(PROBS + 1)]

verbose = 1

def lg(i):
    if i <= 1:
        return 0
    return (i - 1).bit_length()

def pt_off_set(ptp):
    for i in range(PROBS):
        dist = ptp[i + 1].vmin - ptp[i].vmin
        ptp[i].off = lg(dist)

def pt_init(ptp, vmax):
    vstep = vmax // PROBS
    for i in range(PROBS + 1):
        ptp[i].vmin = i * vstep
    pt_off_set(ptp)

def pt_print_final(ptp):
    tbits = abits = obits = vcnt = 0
    print("PT_FINAL: vmin off abits obits ab100 ob100 tb100 vcnt frac")
    for i in range(PROBS + 1):
        tbits += ptp[i].abits + ptp[i].obits
        abits += ptp[i].abits
        obits += ptp[i].obits
        vcnt = ptp[i].vcnt

    for i in range(PROBS + 1):
        print(
            f"[{ptp[i].vmin:3}, {ptp[i].off:2}] : "
            f"{ptp[i].abits:10} {ptp[i].obits:10} "
            f"{ptp[i].abits / tbits:.3f} {ptp[i].obits / tbits:.3f} "
            f"{(ptp[i].abits + ptp[i].obits) / tbits:.3f} "
            f"{ptp[i].vcnt:10} {ptp[i].vcnt / vcnt:.3f}"
        )
    print()

def pt_print(ptp):
    print("PT_INIT:", end=" ")
    for i in range(PROBS + 1):
        print(f"[{ptp[i].vmin}, {ptp[i].off} ({ptp[i].vcnt})]", end=" ")
    print()

def entropy_precision(f):
    f *= (1 << PROB_BITS)
    f = round(f)
    f /= (1 << PROB_BITS)
    return 0 if f == 0 else math.log2(f)

def pt_encoded_size(hist, ptp):
    rcnt = [0] * (PROBS + 1)
    ptotal = ototal = 0

    pt_off_set(ptp)

    for i in range(PROBS):
        ptp[i].obits = 0
        for v in range(ptp[i].vmin, ptp[i + 1].vmin):
            rcnt[i] += hist[v]
            ototal += hist[v] * ptp[i].off
            ptotal += hist[v]
            ptp[i].obits += hist[v] * ptp[i].off
        ptp[i].vcnt = rcnt[i]

    btotal = 0
    for i in range(PROBS):
        prob = rcnt[i] / ptotal
        btotal += rcnt[i] * entropy_precision(prob)
        ptp[i].abits = round(-rcnt[i] * entropy_precision(prob))

    return ototal - btotal

def pt_copy(dest, src):
    for i in range(PROBS + 1):
        dest[i] = Pte(
            vmin=src[i].vmin,
            off=src[i].off,
            abits=src[i].abits,
            obits=src[i].obits,
            vcnt=src[i].vcnt,
        )

def search_try(hist, trial_in, score_best, ptbest, depth, around):
    trial = [Pte() for _ in range(PROBS + 1)]
    pt_copy(trial, trial_in)

    for c in range(1, PROBS):
        if around >= 0 and abs(c - around) != 1:
            continue
        while trial[c].vmin > trial[c - 1].vmin:
            trial[c].vmin -= 1
            if depth < DEPTH_MAX:
                search_try(hist, trial, score_best, ptbest, depth + 1, c)
            else:
                score_new = pt_encoded_size(hist, trial)
                if score_new < score_best[0]:
                    pt_copy(ptbest, trial)
                    score_best[0] = score_new
                    if verbose == 1:
                        print("PTBEST:", end=" ")
                        pt_print(ptbest)
        while trial[c].vmin < trial[c + 1].vmin:
            trial[c].vmin += 1
            if depth < DEPTH_MAX:
                search_try(hist, trial, score_best, ptbest, depth + 1, c)
            else:
                score_new = pt_encoded_size(hist, trial)
                if score_new < score_best[0]:
                    pt_copy(ptbest, trial)
                    score_best[0] = score_new
                    if verbose == 1:
                        print("PTBEST:", end=" ")
                        pt_print(ptbest)

def search(bits, hist, ptab, in_verbose):
    global vmax
    vmax = 1 << bits

    pt_init(pbest, vmax)
    score_raw = score_best = [round(pt_encoded_size(hist, pbest))]

    if verbose == 1:
        pt_print(pbest)

    while True:
        score_best[0] = pt_encoded_size(hist, pbest)
        for i in range(PROBS + 1):
            pnew[i] = Pte(
                vmin=pbest[i].vmin,
                off=pbest[i].off,
                abits=pbest[i].abits,
                obits=pbest[i].obits,
                vcnt=pbest[i].vcnt,
            )
        prev_best = score_best[0]
        if verbose == 1:
            print(f"ENCODED: {score_best[0]:.6f}")
        search_try(hist, pnew, score_best, pbest, 2, -2)
        if verbose == 1:
            print(f"ENCODED: {score_best[0]:.6f}")
        if score_best[0] / prev_best > 0.99:
            break

    if verbose > 1:
        pt_print_final(pbest)
    for i in range(PROBS):
        ptab[i] = pbest[i]
    if verbose:
        pass
    # for i in range(1 << bits):
    #     print(f"{i} {hist[i]}") 
