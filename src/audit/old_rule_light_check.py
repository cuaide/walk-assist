import argparse, csv, os, re, collections
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument('--ds', required=True, help='dataset root with groups.csv')
ap.add_argument('--out', default='audit/old_rule_check')
ap.add_argument('--thr', type=float, default=1.3, help='suspect if h/w is below this value (1/thr in terms of w/h)')
ap.add_argument('--splits', nargs='+', default=['valid', 'test', 'train'])
ap.add_argument('--sheet-max', type=int, default=300, help='max crops per split in the contact sheet')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

import yaml
names = yaml.safe_load(open(f'{a.ds}/data.yaml'))['names']
if isinstance(names, dict): names = [names[k] for k in sorted(names)]
LIGHT = {names.index('green-crossing'): 'green', names.index('red-crossing'): 'red'}

old = {}
gp = f'{a.ds}/groups.csv'
if os.path.exists(gp):
    for r in csv.DictReader(open(gp, encoding='utf-8-sig')):
        old[os.path.splitext(r['file'])[0]] = r.get('old_split', '?') or '?'
else:
    print('groups.csv not found -> skipping old_split comparison')

def prefix(stem):
    for p in ['MP_', 'ZED', 'IMG_', 'KakaoTalk', 'sam_', 'heon', 'john', 'rpl', 'gpl']:
        if stem.startswith(p): return p.rstrip('_')
    return 'numeric' if re.match(r'^\d', stem) else stem.split('_')[0][:10]

def img_path(sp, stem):
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.webp', '.bmp']:
        p = f'{a.ds}/{sp}/images/{stem}{ext}'
        if os.path.exists(p): return p
    return None

boxes = []   # dict per light box
for sp in a.splits:
    ld = f'{a.ds}/{sp}/labels'
    if not os.path.isdir(ld): print(sp, ': label folder not found'); continue
    for lf in sorted(os.listdir(ld)):
        if not lf.endswith('.txt'): continue
        stem = lf[:-4]
        rows = [l.split() for l in open(f'{ld}/{lf}') if l.strip()]
        rows = [r for r in rows if len(r) == 5 and int(float(r[0])) in LIGHT]
        if not rows: continue
        ip = img_path(sp, stem)
        if ip is None: continue
        W, H = Image.open(ip).size
        for c, x, y, w, h in rows:
            x, y, w, h = map(float, (x, y, w, h)); pw, ph = w * W, h * H
            if pw < 1 or ph < 1:
                continue
            edge = (x - w / 2 < 0.003) or (x + w / 2 > 0.997) or (y - h / 2 < 0.003) or (y + h / 2 > 0.997)
            boxes.append(dict(split=sp, stem=stem, img=ip, cls=LIGHT[int(float(c))], x=x, y=y, w=w, h=h,pw=pw, ph=ph, hw=ph / pw, edge=edge, src=prefix(stem),old=old.get(stem, '?')))
print('light boxes', len(boxes))

lines = []
def say(s=''): print(s); lines.append(s)

# 1) Distribution (per old_split) -------------------------------------------
say('== h/w distribution (in pixels, excluding boxes cut off at image edge) ==')
def q(v): return ' '.join(f'{p}%={np.percentile(v, p):.2f}' for p in (5, 10, 25, 50, 75, 90)) if len(v) else '-'
for cls in ['green', 'red']:
    for sp in a.splits:
        for o in ['train', 'valid', 'test', '?']:
            v = [b['hw'] for b in boxes if b['cls'] == cls and b['split'] == sp and b['old'] == o and not b['edge']]
            if v: say(f'{cls:5} new={sp:5} old={o:5} n={len(v):5}  {q(v)}')

# 2) Threshold table: old_train (v2 baseline) vs old_valid_test --------------
say('\n== Ratio of h/w < t: old_train (boxes fixed in v2) vs old_valid/test (possibly unfixed), new valid+test only ==')
say('   t with the largest gap = value that best separates lit-lamp-only boxes from whole-signal-head boxes')
ev = [b for b in boxes if b['split'] in ('valid', 'test') and not b['edge']]
best = {}
for cls in ['green', 'red']:
    A = np.array([b['hw'] for b in ev if b['cls'] == cls and b['old'] == 'train'])
    B = np.array([b['hw'] for b in ev if b['cls'] == cls and b['old'] in ('valid', 'test')])
    if not len(A) or not len(B): say(f'{cls}: no groups to compare (old_train {len(A)}, old_vt {len(B)})'); continue
    say(f'{cls}: old_train n={len(A)}, old_valid/test n={len(B)}')
    gaps = []
    for t in np.arange(0.8, 2.61, 0.1):
        fa, fb = (A < t).mean(), (B < t).mean()
        gaps.append((fb - fa, t))
        say(f'   t={t:.1f}  old_train {fa:6.1%}  old_v/t {fb:6.1%}  gap {fb - fa:+6.1%}  (w/h > {1 / t:.2f})')
    g, t = max(gaps); best[cls] = t
    say(f'   -> {cls} max gap t={t:.1f} ({g:+.1%})')

# 3) Suspect box counts ------------------------------------------------------
sus = [b for b in boxes if b['hw'] < a.thr]
say(f'\n== Suspect boxes: h/w < {a.thr} (= w/h > {1 / a.thr:.2f}) ==  [including edge-cut, parentheses = excluding edge-cut]')
for sp in a.splits:
    for cls in ['green', 'red']:
        tot = sum(1 for b in boxes if b['split'] == sp and b['cls'] == cls)
        s = [b for b in sus if b['split'] == sp and b['cls'] == cls]
        ne = sum(1 for b in s if not b['edge'])
        say(f'{sp:5} {cls:5}: {len(s):4} ({ne}) / {tot}  = {len(s) / max(tot, 1):.1%}   images {len({b["stem"] for b in s})}')
say('\n-- split x source prefix x old_split (valid/test) --')
cnt = collections.Counter((b['split'], b['src'], b['old'], b['cls']) for b in sus if b['split'] in ('valid', 'test'))
for k, v in sorted(cnt.items(), key=lambda kv: (kv[0][0], -kv[1])):
    say(f'{k[0]:5} {k[1]:10} old={k[2]:5} {k[3]:5} {v}')
open(f'{a.out}/summary.txt', 'w', encoding='utf-8').write('\n'.join(lines))

with open(f'{a.out}/suspects.csv', 'w', newline='', encoding='utf-8-sig') as f:
    wr = csv.writer(f)
    wr.writerow(['split', 'file', 'cls', 'src', 'old_split', 'box_w_px', 'box_h_px', 'h_over_w', 'edge_cut'])
    for b in sorted(sus, key=lambda b: (b['split'], b['hw'])):
        wr.writerow([b['split'], os.path.basename(b['img']), b['cls'], b['src'], b['old'],round(b['pw']), round(b['ph']), round(b['hw'], 2), int(b['edge'])])

# 4) contact sheet ----------------------------------------------------------
T, COLS = 160, 8
def crop(b):
    im = Image.open(b['img']).convert('RGB'); W, H = im.size
    x0, y0 = (b['x'] - b['w'] / 2) * W, (b['y'] - b['h'] / 2) * H
    x1, y1 = x0 + b['pw'], y0 + b['ph']
    m = max(b['pw'], b['ph']) * 0.8 + 8
    cx0, cy0, cx1, cy1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    c = im.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
    d = ImageDraw.Draw(c)
    d.rectangle([x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0], outline=(255, 0, 255), width=max(1, int(c.width / 120)))
    c.thumbnail((T, T - 22))
    tile = Image.new('RGB', (T, T), (30, 30, 30)); tile.paste(c, ((T - c.width) // 2, 22))
    td = ImageDraw.Draw(tile)
    td.text((2, 1), f"{b['cls'][0].upper()} h/w {b['hw']:.2f} {b['old'][:2]}{' E' if b['edge'] else ''}", fill=(255, 255, 0))
    td.text((2, 11), b['stem'][:26], fill=(200, 200, 200))
    return tile

for sp in ('valid', 'test'):
    s = sorted([b for b in sus if b['split'] == sp], key=lambda b: (b['cls'], b['hw']))[:a.sheet_max]
    for i in range(0, len(s), 64):
        part = s[i:i + 64]; rows_ = (len(part) + COLS - 1) // COLS
        sheet = Image.new('RGB', (COLS * T, rows_ * T), (0, 0, 0))
        for j, b in enumerate(part): sheet.paste(crop(b), ((j % COLS) * T, (j // COLS) * T))
        sheet.save(f'{a.out}/sheet_{sp}_{i // 64:02d}.jpg', quality=88)
    print(sp, 'sheets:', (len(s) + 63) // 64)
print('\nSaved:', a.out, '(summary.txt, suspects.csv, sheet_*.jpg)')
print('Tile label: G/R, h/w, first 2 chars of old_split (tr/va/te), E=cut off at image edge. Magenta border = current GT box')
