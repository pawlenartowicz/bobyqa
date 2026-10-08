#!/usr/bin/env python3
"""oracle/states.py - split a bobyqa_states.dump stream into "bobyqa state v1" files.

Pairs the i-th entry of each routine with its i-th exit (valid because the dump
sites never recurse), subsamples (see `subsample`), and writes one file
per kept (routine, call) into <outdir>/<routine>/.
"""
import argparse
import sys
from pathlib import Path


def parse_blocks(lines):
    """Yield (routine, marker, body_lines) for each state...end block."""
    blocks, cur = [], None
    for ln in lines:
        t = ln.split()
        if not t:
            continue
        if t[0] == 'state':
            assert cur is None, f'nested state block: {ln}'
            cur = (t[1], t[2], [])
        elif t[0] == 'end':
            assert cur is not None, 'end without state'
            blocks.append(cur)
            cur = None
        elif cur is not None:
            cur[2].append(ln.rstrip('\n'))
    assert cur is None, 'unterminated state block'
    return blocks


def pair(blocks):
    """routine -> [(entry_body, exit_body), ...] in call order."""
    pairs, pending = {}, {}
    for routine, marker, body in blocks:
        if marker == 'entry':
            pending.setdefault(routine, []).append(body)
        elif marker == 'exit':
            ent = pending[routine].pop()  # strict nesting, no recursion
            pairs.setdefault(routine, []).append((ent, body))
        else:
            raise ValueError(f'bad marker {marker!r}')
    leftover = {r: len(v) for r, v in pending.items() if v}
    # An unpaired entry means a routine took an early RETURN past its exit dump:
    # fix instrument.patch (duplicate the exit block before that return).
    assert not leftover, f'unpaired entries (early returns?): {leftover}'
    return pairs


def subsample(n, keep):
    """Indices to keep: all if n <= keep, else a head + an evenly spaced tail."""
    if n <= keep:
        return list(range(n))
    head = 5 if keep >= 10 else 1  # first 5 plus a uniform sample
    rest = keep - head
    if rest == 1:
        tail = [n - 1]
    else:
        tail = [head + ((n - head - 1) * i) // (rest - 1) for i in range(rest)]
    return list(range(head)) + sorted(set(tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump')
    ap.add_argument('--problem', required=True)
    ap.add_argument('--npt', required=True)
    ap.add_argument('--rhobeg', required=True)
    ap.add_argument('--rhoend', required=True)
    ap.add_argument('--maxfun', required=True)
    ap.add_argument('--prima', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--keep', type=int, default=10)
    ap.add_argument('--tag', default='')  # filename disambiguator for stress configs
    a = ap.parse_args()

    pairs = pair(parse_blocks(Path(a.dump).read_text().splitlines()))
    tag = f'_{a.tag}' if a.tag else ''
    for routine, plist in sorted(pairs.items()):
        outdir = Path(a.outdir) / routine
        outdir.mkdir(parents=True, exist_ok=True)
        kept = subsample(len(plist), a.keep)
        for i in kept:
            ent, ext = plist[i]
            path = outdir / f'{a.problem}_npt{a.npt}{tag}_{i + 1:03d}.txt'
            with path.open('w') as out:
                out.write('# bobyqa state v1\n')
                out.write(f'# prima {a.prima}\n')
                out.write(f'routine {routine}\n')
                out.write(f'problem {a.problem}\n')
                out.write(f'npt {a.npt}\n')
                out.write(f'rho_begin {a.rhobeg}\n')
                out.write(f'rho_end {a.rhoend}\n')
                out.write(f'max_fun {a.maxfun}\n')
                out.write(f'seq {i + 1}\n')
                out.write('entry\n')
                out.writelines(l + '\n' for l in ent)
                out.write('exit\n')
                out.writelines(l + '\n' for l in ext)
        print(f'{routine}: kept {len(kept)}/{len(plist)}', file=sys.stderr)


if __name__ == '__main__':
    main()
