import argparse, csv, os, re, collections
import numpy as np, torch, yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

ap = argparse.ArgumentParser()
ap.add_argument('--ds', required=True, help='dataset root (ds_v4)')
ap.add_argument('--weights', required=True)
ap.add_argument('--out', default='audit/light_miss')
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--conf', type=float, default=0.25)
ap.add_argument('--split', default='valid')
a = ap.parse_args()
assert a.split == 'valid', 'do not look at test scores until the threshold/rules are frozen'
os.makedirs(a.out, exist_ok=True)

names = yaml.safe_load(open(f'{a.ds}/data.yaml'))['names']
if isinstance(names, dict): names = [names[k] for k in sorted(names)]
G, R = names.index('green-crossing'), names.index('red-crossing')
LIGHT = {G: 'green', R: 'red'}
old = {}
if os.path.exists(f'{a.ds}/groups.csv'):
    for r in csv.DictReader(open(f'{a.ds}/groups.csv', encoding='utf-8-sig')):
        old[os.path.splitext(r['file'])[0]] = r.get('old_split', '?')

def prefix(s):
    for p in ['MP_', 'ZED', 'IMG_', 'KakaoTalk', 'sam_', 'heon', 'john', 'rpl', 'gpl', 'Bbox']:
        if s.startswith(p): return p.rstrip('_')
    return 'numeric' if re.match(r'^\d', s) else s.split('_')[0][:10]

m = YOLO(a.weights)
rows = []
for r in m.predict(f'{a.ds}/{a.split}/images', imgsz=a.imgsz, conf=0.05, stream=True, verbose=False):
    stem = os.path.splitext(os.path.basename(r.path))[0]
    lf = f'{a.ds}/{a.split}/labels/{stem}.txt'
    if not os.path.exists(lf): continue
    gt = [l.split() for l in open(lf) if l.strip()]
    gt = [list(map(float, g)) for g in gt if len(g) == 5 and int(float(g[0])) in LIGHT]
    if not gt: continue
    H, W = r.orig_shape
    pb, pc, ps = r.boxes.xyxy.cpu(), r.boxes.cls.cpu().int(), r.boxes.conf.cpu()
    keep = (pc == G) | (pc == R)
    pb, pc, ps = pb[keep], pc[keep], ps[keep]
    for c, x, y, w, h in gt:
        c = int(c)
        g = torch.tensor([[(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]])
        cat, best_iou, best_box, best_conf, best_cls = 'miss', 0., None, 0., -1
        if len(pb):
            iou = box_iou(g, pb)[0]
            hi = ps >= a.conf
            if hi.any():
                i = torch.where(hi, iou, torch.tensor(-1.)).argmax()
                best_iou, best_box, best_conf, best_cls = float(iou[i]), pb[i].tolist(), float(ps[i]), int(pc[i])
                if best_iou >= 0.5: cat = 'hit' if best_cls == c else 'wrong_col'
                elif best_iou >= 0.1: cat = 'loc'
            if cat == 'miss' and (iou >= 0.5).any():
                i = torch.where(iou >= 0.5, ps, torch.tensor(-1.)).argmax()
                cat, best_iou, best_box, best_conf, best_cls = 'lowconf', float(iou[i]), pb[i].tolist(), float(ps[i]), int(pc[i])
        rows.append(dict(file=os.path.basename(r.path), img=r.path, cls=LIGHT[c], src=prefix(stem), old=old.get(stem, '?'),
                         gt=g[0].tolist(), h_px=h * H, hw=(h * H) / max(w * W, 1e-6), cat=cat, iou=best_iou,
                         pred=best_box, pconf=best_conf, pcls=LIGHT.get(best_cls, '-')))

CATS = ['hit', 'wrong_col', 'loc', 'lowconf', 'miss']
lines = []
def say(s=''): print(s); lines.append(s)
def table(key):
    say(f'\n== by {key} (conf {a.conf}) ==')
    say(f'{"":28} {"n":>5} ' + ' '.join(f'{c:>9}' for c in CATS) + '   recall')
    grp = collections.defaultdict(list)
    for x in rows: grp[(x['cls'],) + tuple(x[k] for k in key)].append(x['cat'])
    for k in sorted(grp):
        v = grp[k]; n = len(v)
        if n < 5: continue
        say(f'{" ".join(map(str, k)):28} {n:5} ' + ' '.join(f'{v.count(c) / n:9.1%}' for c in CATS) + f'   {v.count("hit") / n:.3f}')
table(())
table(('old',))
table(('src',))
table(('src', 'old'))
# Size: are the misses small?
say('\n== recall by GT height (px) ==')
for cls in ['green', 'red']:
    for lo, hi in [(0, 20), (20, 40), (40, 80), (80, 1e9)]:
        v = [x['cat'] for x in rows if x['cls'] == cls and lo <= x['h_px'] < hi]
        if v: say(f'{cls:5} h {lo:>3}-{hi if hi < 1e9 else "":<4}px n={len(v):4}  recall {v.count("hit") / len(v):.3f}  miss {v.count("miss") / len(v):.1%}  loc {v.count("loc") / len(v):.1%}')
open(f'{a.out}/summary.txt', 'w', encoding='utf-8').write('\n'.join(lines))

with open(f'{a.out}/per_box.csv', 'w', newline='', encoding='utf-8-sig') as f:
    wr = csv.writer(f); wr.writerow(['file', 'cls', 'src', 'old_split', 'cat', 'iou', 'pred_conf', 'pred_cls', 'h_px', 'h_over_w'])
    for x in rows: wr.writerow([x['file'], x['cls'], x['src'], x['old'], x['cat'], round(x['iou'], 3), round(x['pconf'], 3), x['pcls'], round(x['h_px']), round(x['hw'], 2)])

# Sheets: non-hit boxes, per red/green
T, COLS = 170, 8
def tile(x):
    im = Image.open(x['img']).convert('RGB'); W, H = im.size
    bx = [x['gt']] + ([x['pred']] if x['pred'] else [])
    x0, y0 = min(b[0] for b in bx), min(b[1] for b in bx); x1, y1 = max(b[2] for b in bx), max(b[3] for b in bx)
    mg = max(x1 - x0, y1 - y0) * 0.8 + 8
    cx0, cy0 = max(0, x0 - mg), max(0, y0 - mg)
    c = im.crop((int(cx0), int(cy0), int(min(W, x1 + mg)), int(min(H, y1 + mg)))); d = ImageDraw.Draw(c)
    lw = max(1, int(c.width / 110))
    d.rectangle([x['gt'][0] - cx0, x['gt'][1] - cy0, x['gt'][2] - cx0, x['gt'][3] - cy0], outline=(255, 0, 255), width=lw)
    if x['pred']: d.rectangle([x['pred'][0] - cx0, x['pred'][1] - cy0, x['pred'][2] - cx0, x['pred'][3] - cy0], outline=(0, 220, 255), width=lw)
    c.thumbnail((T, T - 22)); t = Image.new('RGB', (T, T), (30, 30, 30)); t.paste(c, ((T - c.width) // 2, 22))
    td = ImageDraw.Draw(t)
    td.text((2, 1), f"{x['cls'][0].upper()} {x['cat']} iou{x['iou']:.2f} {x['pcls'][0]}{x['pconf']:.2f} {x['old'][:2]}", fill=(255, 255, 0))
    td.text((2, 11), f"{x['h_px']:.0f}px {x['file'][:22]}", fill=(200, 200, 200))
    return t
for cls in ['red', 'green']:
    s = sorted([x for x in rows if x['cls'] == cls and x['cat'] != 'hit'], key=lambda x: (x['cat'], x['src']))[:192]
    for i in range(0, len(s), 64):
        part = s[i:i + 64]; sh = Image.new('RGB', (COLS * T, ((len(part) + COLS - 1) // COLS) * T))
        for j, x in enumerate(part): sh.paste(tile(x), ((j % COLS) * T, (j // COLS) * T))
        sh.save(f'{a.out}/miss_{cls}_{i // 64:02d}.jpg', quality=88)
print('\nSaved:', a.out, '| magenta=GT, cyan=prediction. Label: category, IoU, predicted color+conf, old_split')
