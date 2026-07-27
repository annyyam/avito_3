#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r'[A-Za-zА-Яа-яЁё0-9]+', re.UNICODE)


def tok(s):
    return [
        x.lower()
        for x in TOKEN_RE.findall(s or '')
        if len(x) > 1
    ]


def load_jsonl(path, key):
    d = {}

    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                d[r[key]] = r

    return d


def score(query, docs, k1=1.5, b=0.75):
    td = [tok(x) for x in docs]
    q = tok(query)

    n = len(td)
    avg = sum(map(len, td)) / max(n, 1)

    df = Counter()

    for d in td:
        df.update(set(d))

    ans = []

    for d in td:
        tf = Counter(d)
        dl = len(d)
        s = 0.0

        for term in q:
            f = tf.get(term, 0)

            if not f:
                continue

            idf = math.log(
                1 + (n - df[term] + 0.5) / (df[term] + 0.5)
            )

            s += (
                idf * f * (k1 + 1)
                / (
                    f
                    + k1
                    * (
                        1
                        - b
                        + b * dl / max(avg, 1e-9)
                    )
                )
            )

        ans.append(s)

    return ans


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--needs',
        type=Path,
        default=Path('needs.validation.jsonl')
    )

    p.add_argument(
        '--providers',
        type=Path,
        default=Path('providers.jsonl')
    )

    p.add_argument(
        '--pools',
        type=Path,
        default=Path('candidate_pools.validation.csv')
    )

    p.add_argument(
        '--output',
        type=Path,
        default=Path('submission.bm25.validation.jsonl')
    )

    a = p.parse_args()

    needs = load_jsonl(a.needs, 'need_id')
    providers = load_jsonl(a.providers, 'candidate_id')

    pools = defaultdict(list)

    with a.pools.open(
        encoding='utf-8-sig',
        newline=''
    ) as f:
        for r in csv.DictReader(f):
            pools[r['need_id']].append(
                r['candidate_id']
            )

    with a.output.open(
        'w',
        encoding='utf-8'
    ) as out:
        for nid in sorted(pools):
            ids = pools[nid]

            docs = [
                providers[c]['profile_summary']
                for c in ids
            ]

            scores = score(
                needs[nid]['need_text'],
                docs
            )

            ranked = sorted(
                zip(ids, scores),
                key=lambda x: (-x[1], x[0])
            )[:5]

            result = {
                'need_id': nid,
                'ranked_candidates': [
                    {
                        'candidate_id': c,
                        'score': s
                    }
                    for c, s in ranked
                ]
            }

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + '\n'
            )


if __name__ == '__main__':
    main()
