import argparse, os, re, glob, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', required=True, help='dataset with valid/ and test/ images')
ap.add_argument('--audit-dir', default='audit', help='folder with exclude_<split>.csv and cutout_<split>.csv; subset lists are written here')
a = ap.parse_args()
DS, OUT_DIR = a.data_root, a.audit_dir

def base(fname):                                   # same rule as resplit_groups.clean_base
    b = os.path.splitext(fname)[0].split('.rf.')[0]
    b = re.sub(r'_augmented_\d+', '', b)
    b = re.sub(r'(_(jpe?g|png|JPE?G|PNG))+$', '', b)
    return re.sub(r'^Bbox_\d+__', '', b)

def stem(fname):                                   # same rule as leak_followup.stem
    s = os.path.basename(fname).split('.rf.')[0]
    s = re.sub(r'_(jpe?g|png|JPE?G|PNG)$', '', s)
    return re.sub(r'_augmented_\d+$', '', s)

for sp in ['valid', 'test']:
    files = sorted(os.path.basename(f) for f in glob.glob(f'{DS}/{sp}/images/*'))
    seq = set(pd.read_csv(f'{OUT_DIR}/exclude_{sp}.csv').img) if os.path.exists(f'{OUT_DIR}/exclude_{sp}.csv') else set()
    cut = set(pd.read_csv(f'{OUT_DIR}/cutout_{sp}.csv').image) if os.path.exists(f'{OUT_DIR}/cutout_{sp}.csv') else set()

    drop_aug = set(cut)
    by_base = {}
    for f in files: by_base.setdefault(base(f), []).append(f)
    n_multi = 0
    for b, fs in by_base.items():
        if len(fs) < 2: continue
        n_multi += 1
        keep = sorted(fs, key=lambda f: (f in cut, f))[0]   # prefer a copy without cutout
        drop_aug |= {f for f in fs if f != keep}

    subsets = {
        'full':   files,
        'no_seq': [f for f in files if stem(f) not in seq],
        'no_aug': [f for f in files if f not in drop_aug],
    }
    subsets['clean'] = [f for f in subsets['no_seq'] if f not in drop_aug]
    # Korea-only (target use): AI Hub MP/ZED + own photos/videos. Fixed 11/20 before test.
    subsets['korea'] = [f for f in files if re.match(r'^(MP_|ZED|IMG_|KakaoTalk_)', base(f))]
    for name, lst in subsets.items():
        open(f'{OUT_DIR}/subset_{sp}_{name}.txt', 'w').write('\n'.join(lst) + '\n')
    print(f'{sp:5}: ' + ', '.join(f'{k} {len(v)}' for k, v in subsets.items())
          + f'  | cutout {len(cut)}, originals with >1 copy {n_multi}')
