#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path

def dcg(grades,k=5): return sum((2**g-1)/math.log2(i+2) for i,g in enumerate(grades[:k]))
def load_labels(path):
    d={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            d[(r['need_id'],r['candidate_id'])]=(int(r['relevance_grade']),str(r['hard_eligible']).lower()=='true')
    return d
def load_pools(path):
    p=defaultdict(set)
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f): p[r['need_id']].add(r['candidate_id'])
    return p
def main():
    ap=argparse.ArgumentParser(description='Local NDCG@5 evaluator. Use train labels for local checks; validation/test gold is organizer-only.')
    ap.add_argument('--submission',type=Path,required=True)
    ap.add_argument('--labels',type=Path,default=Path('labels.train.csv'))
    ap.add_argument('--pools',type=Path,default=Path('candidate_pools.train.csv'))
    a=ap.parse_args(); labels=load_labels(a.labels); pools=load_pools(a.pools)
    submissions=[]
    with a.submission.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): submissions.append(json.loads(line))
    seen=set(); ndcgs=[]; violations=0; total=0
    expected=set(pools); actual={r['need_id'] for r in submissions}
    if actual!=expected: raise SystemExit(f'need coverage mismatch: missing={len(expected-actual)} extra={len(actual-expected)}')
    by_need_labels=defaultdict(list)
    for (nid,cid),(grade,eligible) in labels.items(): by_need_labels[nid].append(grade)
    for row in submissions:
        nid=row['need_id']; ranked=row['ranked_candidates']
        if nid in seen:
            raise SystemExit(f"{nid}: duplicate need_id in submission")
        seen.add(nid)        
        if len(ranked) != 5: raise SystemExit(f'{nid}: expected exactly 5 candidates')
        ids=[x['candidate_id'] for x in ranked]
        if len(ids)!=len(set(ids)): raise SystemExit(f'{nid}: duplicate candidate')
        if not set(ids)<=pools[nid]: raise SystemExit(f'{nid}: candidate outside pool')
        if any(not math.isfinite(float(x['score'])) for x in ranked): raise SystemExit(f'{nid}: non-finite score')
        ranked=sorted(ranked,key=lambda x:(-float(x['score']),x['candidate_id']))
        grades=[]
        for x in ranked:
            key=(nid,x['candidate_id'])
            if key not in labels: raise SystemExit(f'no label for {key}')
            grade,eligible=labels[key]; grades.append(grade); total+=1; violations += (not eligible)
        ideal=sorted(by_need_labels[nid],reverse=True)[:5]
        denom=dcg(ideal)
        ndcgs.append(dcg(grades)/denom if denom else 0.0)
    print(f'NDCG@5: {sum(ndcgs)/len(ndcgs):.6f}')
    print(f'HardConstraintViolationRate@5: {violations/total:.6f}')
    print(f'needs: {len(ndcgs)}')
if __name__=='__main__': main()
