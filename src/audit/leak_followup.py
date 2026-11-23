import argparse, os, re, glob, bisect, collections
import numpy as np, pandas as pd, cv2

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', required=True, help='dataset with train/valid/test/images')
ap.add_argument('--out', default='audit', help='folder for exclude_<split>.csv, leak_family_counts.csv and sheets')
a = ap.parse_args()
SRC, OUT_DIR = a.data_root, a.out
SPLITS  = ['train', 'valid', 'test']
GAP_SEQ = {'rpl': 3, 'gpl': 3, 'IMG': 50}
GAP_DEFAULT = 30
KAKAO_MIN = 15

os.makedirs(OUT_DIR, exist_ok=True)

def stem(f):
    s = os.path.basename(f).split('.rf.')[0]
    s = re.sub(r'_(jpe?g|png|JPE?G|PNG)$', '', s)
    return re.sub(r'_augmented_\d+$', '', s)

def family(s):
    m = re.match(r'[A-Za-z]+', s)
    return m.group() if m else 'numeric'

files = {sp: sorted(glob.glob(f'{SRC}/{sp}/images/*')) for sp in SPLITS}
path = {(sp, stem(f)): f for sp in SPLITS for f in files[sp]}
print('Images per split:', {sp: len(v) for sp, v in files.items()})

# ---- 1) frame-number neighbours in train (min gap over all train images with the same name pattern) ----
train_seq = collections.defaultdict(list)    # (pattern, run index) -> sorted train numbers
train_name = {}
for f in files['train']:
    s = stem(f)
    for i, m in enumerate(re.finditer(r'\d+', s)):
        key = (s[:m.start()] + '#' + s[m.end():], i)
        if key[0].strip('#_- ') == '': continue
        train_seq[key].append(int(m.group())); train_name[(key, int(m.group()))] = s
for k in train_seq: train_seq[k].sort()

flags = []                                     # (split, img, family, reason, gap, train_match)
for sp in ['valid', 'test']:
    for f in files[sp]:
        s, fam = stem(f), family(stem(f))
        best = None
        for i, m in enumerate(re.finditer(r'\d+', s)):
            key = (s[:m.start()] + '#' + s[m.end():], i)
            lst = train_seq.get(key)
            if not lst: continue
            n = int(m.group()); j = bisect.bisect_left(lst, n)
            for c in lst[max(j - 1, 0):j + 1]:
                g = abs(c - n)
                if best is None or g < best[0]: best = (g, train_name[(key, c)])
        if best and best[0] <= GAP_SEQ.get(fam, GAP_DEFAULT):
            flags.append((sp, s, fam, 'frame_gap', best[0], best[1]))

# ---- 2) KakaoTalk clips recorded on the same walk (same date, start time within KAKAO_MIN) ----
def kakao_time(s):
    m = re.match(r'KakaoTalk_(\d{8})_(\d{2})(\d{2})(\d{2})', s)
    return (m.group(1), int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 60) if m else None
train_kakao = collections.defaultdict(list)
for f in files['train']:
    t = kakao_time(stem(f))
    if t: train_kakao[t[0]].append((t[1], stem(f)))
for sp in ['valid', 'test']:
    for f in files[sp]:
        s = stem(f); t = kakao_time(s)
        if not t or not train_kakao.get(t[0]): continue
        d, other = min((abs(m - t[1]), o) for m, o in train_kakao[t[0]])
        if d <= KAKAO_MIN: flags.append((sp, s, 'KakaoTalk', 'same_walk', round(d, 1), other))

# ---- 3) dHash near-duplicates from data_audit_v1.py ----
dh = f'{OUT_DIR}/leak_dhash.csv'     # from data_audit_v1.py on the SAME split; skipped if absent
near = pd.read_csv(dh) if os.path.exists(dh) else pd.DataFrame(columns=['split', 'img', 'hamming', 'train_match'])
for _, r in near.iterrows():
    flags.append((r.split, r.img, family(r.img), 'dhash', int(r.hamming), r.train_match))

F = pd.DataFrame(flags, columns=['split', 'img', 'family', 'reason', 'gap', 'train_match'])

# ---- summary ----
tot = pd.DataFrame([(sp, family(stem(f))) for sp in ['valid', 'test'] for f in files[sp]], columns=['split', 'family'])
summ = (F.drop_duplicates(['split', 'img']).groupby(['split', 'family']).size().rename('flagged').to_frame().join(tot.groupby(['split', 'family']).size().rename('total'), how='outer').fillna(0).astype(int))
summ['pct'] = (100 * summ.flagged / summ.total).round(1)
print('\nvalid/test images with a likely train sibling, by source family:')
print(summ[summ.flagged > 0].to_string())
print('\nBy reason (images can have several):', F.groupby(['split', 'reason']).size().to_dict())
print('\nGap distribution for frame_gap flags:')
print(F[F.reason == 'frame_gap'].groupby('family').gap.describe()[['count', 'min', '50%', 'max']].to_string())
summ.to_csv(f'{OUT_DIR}/leak_family_counts.csv')

# ---- contact sheets: smallest-gap pairs per family, to judge by eye whether they are the same scene ----
def tile(sp, img, match, label):
    a, b = cv2.imread(path.get((sp, img), '')), cv2.imread(path.get(('train', match), ''))
    if a is None or b is None: return None
    t = np.hstack([cv2.resize(a, (256, 256)), cv2.resize(b, (256, 256))])
    cv2.putText(t, label, (4, 18), 0, 0.45, (0, 255, 255), 2)
    return t
for fam, g in F[F.reason != 'dhash'].groupby('family'):
    ts = [tile(r.split, r.img, r.train_match, f'{r.split} {r.img[:18]} | {r.train_match[:18]} d={r.gap}') for _, r in g.sort_values('gap').head(12).iterrows()]
    ts = [t for t in ts if t is not None]
    if not ts: continue
    while len(ts) % 3: ts.append(np.zeros_like(ts[0]))
    cv2.imwrite(f'{OUT_DIR}/leak_seq_{fam}.jpg', np.vstack([np.hstack(ts[i:i+3]) for i in range(0, len(ts), 3)]))

for sp in ['valid', 'test']:
    e = F[F.split == sp].groupby('img').agg(reason=('reason', lambda x: '+'.join(sorted(set(x)))), family=('family', 'first'), gap=('gap', 'min'), train_match=('train_match', 'first')).reset_index()
    e.to_csv(f'{OUT_DIR}/exclude_{sp}.csv', index=False)
    print(f'\n{sp}: {len(e)} / {len(files[sp])} images flagged ({100*len(e)/len(files[sp]):.1f}%) -> exclude_{sp}.csv')
print('\nSheets:', sorted(os.path.basename(p) for p in glob.glob(f'{OUT_DIR}/leak_seq_*.jpg')))
