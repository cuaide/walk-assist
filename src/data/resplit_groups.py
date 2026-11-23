import argparse, csv, json, random, re, shutil, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image

SPLITS = ('train', 'valid', 'test')
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def clean_base(name):
    base = Path(name).stem.split('.rf.')[0]
    base = re.sub(r'_augmented_\d+', '', base)
    base = re.sub(r'(_(jpe?g|png|JPE?G|PNG))+$', '', base)
    base = re.sub(r'^Bbox_\d+__', '', base)
    return base

def seq_id(name):
    base = clean_base(name)
    m = re.match(r'^((?:rpl|gpl)-)(\d+)', base)
    if m:
        if m.group(1) == 'rpl-' and 900 <= int(m.group(2)) < 1300:
            return 'rpl-scene-900', None
        return m.group(1), int(m.group(2))
    m = re.match(r'^KakaoTalk_(\d{8})_(\d{2})(\d{2})', base)
    if m:
        return f'KakaoTalk_{m.group(1)}_min', int(m.group(2)) * 60 + int(m.group(3))
    m = re.match(r'^(ZED\d_[A-Z]+)_(\d+)', base)
    if m:
        return m.group(1), int(m.group(2))
    m = re.match(r'^(.*?)(\d+)\D*$', base)
    if m and (re.search(r'[A-Za-z]', m.group(1)) or len(m.group(1).strip('_-')) >= 6):
        return m.group(1), int(m.group(2))
    return base, None

def small_gray(path):
    if cv2 is not None:
        im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        return cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
    try:
        return np.asarray(Image.open(path).convert('L').resize((64, 64), Image.BOX), dtype=np.uint8)
    except Exception:
        return None


def resize9x8(g):
    if cv2 is not None:
        return cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(g).resize((9, 8), Image.BOX))


def load_features(items, cache_path):
    feat = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        feat = dict(zip(z['keys'].tolist(), z['sm']))
    todo = [it for it in items if it['key'] not in feat]
    t0 = time.time()
    for n, it in enumerate(todo, 1):
        g = small_gray(it['img'])
        feat[it['key']] = g if g is not None else np.zeros((64, 64), np.uint8)
        if n % 500 == 0:
            print(f'  image features {n}/{len(todo)} ({time.time() - t0:.0f}s)', flush=True)
    if todo:
        try:
            np.savez(cache_path, keys=np.array(list(feat.keys()), dtype=object), sm=np.array(list(feat.values())))
        except OSError as e:
            print('cache save failed (ignored):', e)
    return feat


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[a] = b


def near_pairs(items, feat, hamming, corr):
    N = len(items)
    SM = np.stack([feat[it['key']] for it in items]).astype(np.float32)
    D = np.stack([(lambda s: (s[:, 1:] > s[:, :-1]).flatten())(resize9x8(feat[it['key']])) for it in items])
    D = np.packbits(D, axis=1)                                  # (N, 8) uint8
    pop = np.array([bin(i).count('1') for i in range(256)], np.uint8)
    Z = SM.reshape(N, -1)
    Z = Z - Z.mean(1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    blank = SM.reshape(N, -1).std(1) < 2                         # skip unreadable / single-colour images
    pairs = []
    for i in range(N - 1):
        if blank[i]:
            continue
        dd = pop[np.bitwise_xor(D[i + 1:], D[i])].sum(1)
        for j in np.where(dd <= hamming)[0] + i + 1:
            if blank[j]:
                continue
            c = float(Z[i] @ Z[j])
            if c > corr:
                pairs.append((i, int(j), int(dd[j - i - 1]), round(c, 4)))
    return pairs

def split_groups(groups, items, ratios, seeds, iters, nc, fams=None):
    keys = list(groups)
    fam_idx = {f: j for j, f in enumerate(fams or [])}
    D = nc + 1 + len(fam_idx)
    vec = {}
    for k in keys:
        v = np.zeros(D)
        v[nc] = len(groups[k])
        for i in groups[k]:
            for c in items[i]['classes']:
                if c < nc:
                    v[c] += 1
            if fam_idx:
                v[nc + 1 + fam_idx[items[i]['fam']]] += 1
        vec[k] = v
    tot = sum(vec.values())
    tgt = {s: r * tot for s, r in zip(SPLITS, ratios)}
    forced = {k for k in keys if any(items[i]['aug'] for i in groups[k])}

    wgt = np.ones(D)
    if fam_idx:
        wgt[nc] = 0

    def err(cur):
        return sum(float((np.abs((cur[s] - tgt[s]) / np.maximum(tot, 1)) * wgt).sum()) for s in SPLITS)

    def assign(seed):
        rnd = random.Random(seed)
        so, size = {}, Counter()
        for k in forced:
            so[k] = 'train'; size['train'] += len(groups[k])
        rest = [k for k in keys if k not in forced]
        rnd.shuffle(rest)
        rest.sort(key=lambda k: -len(groups[k]))
        target = {s: r * len(items) for s, r in zip(SPLITS, ratios)}
        for k in rest:
            s = max(SPLITS, key=lambda s: target[s] - size[s])
            so[k] = s; size[s] += len(groups[k])
        return so

    def cur_of(so):
        return {s: sum((vec[k] for k in keys if so[k] == s), np.zeros(D)) for s in SPLITS}

    best = min(range(seeds), key=lambda sd: err(cur_of(assign(sd))))
    so = assign(best)
    cur = cur_of(so)
    e0 = e = err(cur)
    rnd = random.Random(0)
    movable = [k for k in keys if k not in forced]
    for _ in range(iters if movable else 0):
        k = rnd.choice(movable)
        s0 = so[k]
        s1 = rnd.choice([x for x in SPLITS if x != s0])
        cur[s0] -= vec[k]; cur[s1] += vec[k]
        e1 = err(cur)
        if e1 < e:
            e = e1; so[k] = s1
        else:
            cur[s0] += vec[k]; cur[s1] -= vec[k]
    return so, best, (round(e0, 4), round(e, 4))

def read_names(src):
    y = src / 'data.yaml'
    if not y.exists():
        return {}
    t = y.read_text(encoding='utf-8')
    m = re.search(r'names:\s*(\[.*?\])', t, re.S)
    if m:
        return {i: n.strip(" '\"") for i, n in enumerate(m.group(1).strip('[]').split(','))}
    m = re.search(r'names:\s*\n((?:\s+\d+:.*\n?)+)', t)     # 'names:\n  0: crosswalk' format
    if m:
        return {int(a): b.strip(" '\"") for a, b in re.findall(r'(\d+):\s*(.+)', m.group(1))}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', help='folder with train/valid/test and data.yaml')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ratios', default='0.7,0.15,0.15')
    ap.add_argument('--gap', type=int, default=10, help='files whose numbers differ by at most this are grouped')
    ap.add_argument('--hamming', type=int, default=6, help='max dHash Hamming distance for near-identical scenes (0-64)')
    ap.add_argument('--corr', type=float, default=0.93, help='min grayscale correlation for near-identical scenes')
    ap.add_argument('--no-hash', action='store_true', help='disable image-hash merging (name rules only)')
    ap.add_argument('--seeds', type=int, default=200)
    ap.add_argument('--iters', type=int, default=20000, help='number of local-search iterations')
    ap.add_argument('--stratify-source', action='store_true',
                    help='also match the train/valid/test ratio within each source (file-name prefix: sam, MP, ZED, IMG ...) (added 11/20)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    ratios = [float(x) for x in a.ratios.split(',')]
    names = read_names(src)

    # 1) read
    items = []
    for s in SPLITS:
        idir = src / s / 'images'
        if not idir.exists():
            continue
        for img in sorted(idir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lab = src / s / 'labels' / (img.stem + '.txt')
            classes = []
            if lab.exists():
                classes = [int(float(l.split()[0])) for l in lab.read_text().splitlines() if l.strip()]
            fm = re.match(r'[A-Za-z]+', clean_base(img.name))
            items.append(dict(name=img.name, img=img, lab=lab if lab.exists() else None, classes=classes,
                              aug='_augmented_' in img.name, old=s, key=f'{s}/{img.name}',
                              fam=fm.group() if fm else 'numeric'))
    if not items:
        sys.exit(f'no images found: check that {src}/train/images etc. exist')
    N = len(items)
    nc = max(len(names), 1 + max((c for it in items for c in it['classes']), default=0))
    if not names:
        names = {i: str(i) for i in range(nc)}
    dup_names = [n for n, c in Counter(it['name'] for it in items).items() if c > 1]
    print(f'read {N} images (classes {[names[i] for i in range(nc)]})')
    if dup_names:
        print(f'⚠ {len(dup_names)} file names appear in more than one split (e.g. {dup_names[:3]}) -> put in the same group')

    # 2) name-rule groups
    dsu = DSU(N)
    by_prefix = defaultdict(list)
    for i, it in enumerate(items):
        p, n = seq_id(it['name'])
        by_prefix[(p, n is None)].append((n, i))
    for (p, single), lst in by_prefix.items():
        if single:
            for _, i in lst[1:]:
                dsu.union(i, lst[0][1])
            continue
        lst.sort(key=lambda t: t[0])
        prev_n, prev_i = lst[0]
        for n, i in lst[1:]:
            if n - prev_n <= a.gap:
                dsu.union(i, prev_i)
            prev_n, prev_i = n, i
    by_name = defaultdict(list)
    for i, it in enumerate(items):
        by_name[it['name']].append(i)
    for lst in by_name.values():
        for i in lst[1:]:
            dsu.union(i, lst[0])
    n_name_groups = len({dsu.find(i) for i in range(N)})

    # 3) merge near-identical scenes + diagnose leakage in the input split
    rep = dict(images=N, name_groups=n_name_groups)
    pairs = []
    if not a.no_hash:
        print('computing image features (slow only the first time)...', flush=True)
        feat = load_features(items, src / '_hash_cache.npz')
        t0 = time.time()
        pairs = near_pairs(items, feat, a.hamming, a.corr)
        print(f'near-identical pairs: {len(pairs)} ({time.time() - t0:.0f}s)')
        before = [p for p in pairs if items[p[0]]['old'] != items[p[1]]['old']]
        nd = DSU(N)
        for i, j, _, _ in pairs:
            nd.union(i, j)
        clusters = defaultdict(set)
        for i in range(N):
            clusters[nd.find(i)].add(items[i]['old'])
        rep['input_leak_pairs'] = len(before)
        rep['input_leak_clusters'] = sum(len(v) > 1 for v in clusters.values())
        print(f'\n[current split leakage] near-identical pairs crossing splits: {len(before)}, '
              f'clusters: {rep["input_leak_clusters"]}  <- if this is not 0, current scores are inflated')
        for i, j, d, c in pairs:
            dsu.union(i, j)
    groups = defaultdict(list)
    for i in range(N):
        groups[dsu.find(i)].append(i)
    # group name: prefix of the first file in the group
    gname = {}
    for r, v in groups.items():
        gname[r] = clean_base(items[min(v, key=lambda i: items[i]['name'])]['name'])
    rep['groups'] = len(groups)
    singles = sum(len(v) == 1 for v in groups.values())
    print(f'\n{len(groups)} groups ({n_name_groups} from name rules only -> after hash merge), single-image groups: {singles}')
    big = sorted(((len(v), gname[r]) for r, v in groups.items()), reverse=True)[:8]
    print('largest groups:', big)
    if big and big[0][0] > 0.1 * N:
        print(f'⚠ one group holds {big[0][0] / N:.0%} of all images -> hashing may have merged too much. '
              f'try again with --hamming 4 --corr 0.96')

    # 4) split
    fams = sorted({it['fam'] for it in items}) if a.stratify_source else None
    so, seed, e = split_groups(groups, items, ratios, a.seeds, a.iters, nc, fams)
    rep['seed'] = seed
    rep['split_error_before_after_local_search'] = e
    stats, grp_with = {}, {}
    print(f'\n{"split":6} {"images":>6} {"groups":>5}  ' + '  '.join(f'{names[c]:>14}' for c in range(nc)))
    for s in SPLITS:
        gs = [r for r in groups if so[r] == s]
        its = [i for r in gs for i in groups[r]]
        c = Counter(x for i in its for x in items[i]['classes'])
        stats[s] = dict(images=len(its), groups=len(gs), **{names[k]: c[k] for k in range(nc)})
        grp_with[s] = {names[k]: sum(any(k in items[i]['classes'] for i in groups[r]) for r in gs) for k in range(nc)}
        print(f'{s:6} {len(its):7d} {len(gs):5d}  ' + '  '.join(f'{c[k]:>14}' for k in range(nc)))
    rep['splits'] = stats
    rep['groups_containing_class'] = grp_with
    print('\n[evaluation reliability] distinct groups containing each class (matters more than box count; for signals >= 50 is safe, < 20 is risky)')
    for s in ('valid', 'test'):
        print(f'  {s}: ' + ', '.join(f'{k} {v}' for k, v in grp_with[s].items()))
    fam_tab = defaultdict(Counter)
    for r, v in groups.items():
        for i in v:
            fam_tab[items[i]['fam']][so[r]] += 1
    big_f = sorted(fam_tab, key=lambda f: -sum(fam_tab[f].values()))[:12]
    print('\n[split by source] (train / valid / test, train share)')
    for f in big_f:
        t = fam_tab[f]; n = sum(t.values())
        print(f'  {f:12} {t["train"]:5d} / {t["valid"]:4d} / {t["test"]:4d}   train {t["train"] / n:.0%}')
    rep['source_split'] = {f: dict(fam_tab[f]) for f in fam_tab}
    moved = sum(1 for r, v in groups.items() for i in v if items[i]['old'] != so[r])
    rep['moved_images'] = moved
    leak_after = sum(1 for i, j, _, _ in pairs if so[dsu.find(i)] != so[dsu.find(j)])
    rep['leak_pairs_after'] = leak_after
    print(f'\nimages moved from their original split: {moved} / leak pairs in the new split: {leak_after} (should be 0)')

    if a.dry_run:
        print('\n--dry-run: nothing copied. If the numbers look fine, run again without --dry-run')
        return
    if out.exists():
        sys.exit(f'{out} already exists. Delete it or use another --out (no overwriting)')
    for s in SPLITS:
        (out / s / 'images').mkdir(parents=True)
        (out / s / 'labels').mkdir(parents=True)
    rows = []
    t0 = time.time()
    for n, (r, v) in enumerate(groups.items()):
        s = so[r]
        for i in v:
            it = items[i]
            dst_name = it['name'] if it['name'] not in dup_names else f'{it["old"]}__{it["name"]}'
            shutil.copy2(it['img'], out / s / 'images' / dst_name)
            if it['lab']:
                shutil.copy2(it['lab'], out / s / 'labels' / (Path(dst_name).stem + '.txt'))
            rows.append([dst_name, gname[r], s, it['old']])
        if n % 200 == 0:
            print(f'  copied {len(rows)}/{N} ({time.time() - t0:.0f}s)', flush=True)
    with open(out / 'groups.csv', 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows([['file', 'group', 'split', 'old_split']] + sorted(rows, key=lambda r: (r[1], r[0])))
    with open(out / 'leak_before.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['old_split_a', 'file_a', 'old_split_b', 'file_b', 'dhash_dist', 'corr'])
        for i, j, d, c in pairs:
            if items[i]['old'] != items[j]['old']:
                w.writerow([items[i]['old'], items[i]['name'], items[j]['old'], items[j]['name'], d, c])
    nl = '\n'
    (out / 'data.yaml').write_text(
        f'path: {out.resolve().as_posix()}{nl}train: train/images{nl}val: valid/images{nl}test: test/images{nl}{nl}'
        f'nc: {nc}{nl}names: {[names[i] for i in range(nc)]}{nl}', encoding='utf-8')
    (out / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    print('done:', out)


if __name__ == '__main__':
    main()
