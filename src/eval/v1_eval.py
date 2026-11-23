import argparse, csv, json, os, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def read_names(root):
    t = (root / 'data.yaml').read_text(encoding='utf-8')
    m = re.search(r'names:\s*(\[.*?\])', t, re.S)
    if m:
        return [n.strip(" '\"") for n in m.group(1).strip('[]').split(',')]
    m = re.findall(r'^\s*-\s*(.+)$', t.split('names:')[1], re.M)          # names:\n- a\n- b
    if m:
        return [n.strip(" '\"") for n in m]
    m = re.findall(r'^\s*(\d+):\s*(.+)$', t, re.M)
    return [n.strip(" '\"") for _, n in sorted(m, key=lambda x: int(x[0]))]


def load_manifest(path):
    if not path or not Path(path).exists():
        return {}
    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols = list(rows[0].keys())
    src_col = next((c for c in cols if 'source' in c.lower()), None)
    file_col = max(cols, key=lambda c: sum(str(r[c]).lower().endswith(IMG_EXT) for r in rows[:200]))
    if not src_col:
        print('no source column in manifest:', cols); return {}
    return {Path(r[file_col]).name: r[src_col] for r in rows}


def iou(a, b):
    if len(b) == 0:
        return np.zeros(0)
    ix = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def judge(gts, preds, names, iou_thr=0.5, near_frac=0.15):
    CW = names.index('crosswalk'); G = names.index('green-crossing'); R = names.index('red-crossing')
    P = np.array([p[2:] for p in preds], float).reshape(-1, 4)
    pc = np.array([p[0] for p in preds], int)
    ps = np.array([p[1] for p in preds], float)
    out = []
    for c, *box in gts:
        box = np.array(box, float)
        ious = iou(box, P)
        if c in (G, R):
            light = (pc == G) | (pc == R)
            cand = (ious >= iou_thr) & light
            best_any = float(ious[light].max()) if light.any() else 0.0
            if cand.any():
                k = np.where(cand, ps, -1).argmax()
                res = 'ok' if pc[k] == c else 'wrong_color'
                pred_cls = names[pc[k]]
            else:
                res, pred_cls = 'miss', ''
            out.append(dict(cls=names[c], result=res, pred=pred_cls, best_iou=round(best_any, 3)))
        elif c == CW:
            cand = (ious >= iou_thr) & (pc == CW)
            out.append(dict(cls='crosswalk', result='ok' if cand.any() else 'miss',
                            best_iou=round(float(ious[pc == CW].max()) if (pc == CW).any() else 0.0, 3), pred=''))
        else:
            out.append(dict(cls=names[c], result='skip', pred='', best_iou=0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--manifest', default='')
    ap.add_argument('--split', default='test')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sheet-max', type=int, default=240, help='max number of missed-signal crops in the sheets')
    a = ap.parse_args()
    import cv2
    from ultralytics import YOLO

    root, out = Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = read_names(root)
    src_of = load_manifest(a.manifest)
    G, R = names.index('green-crossing'), names.index('red-crossing')
    model = YOLO(a.weights)
    idir = root / a.split / 'images'
    lights, cws = [], []
    per_src = defaultdict(Counter)
    for r in model.predict(str(idir), imgsz=a.imgsz, conf=a.conf, stream=True, verbose=False):
        fn = Path(r.path).name
        H, W = r.orig_shape
        lf = root / a.split / 'labels' / (Path(fn).stem + '.txt')
        gts = []
        if lf.exists():
            for line in lf.read_text().splitlines():
                v = line.split()
                if len(v) == 5:
                    c, x, y, w, h = int(float(v[0])), *map(float, v[1:])
                    gts.append((c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
        b = r.boxes
        preds = [(int(c), float(s), *map(float, xy)) for c, s, xy in zip(b.cls.cpu().numpy(), b.conf.cpu().numpy(), b.xyxy.cpu().numpy())]
        src = src_of.get(fn, 'unknown')
        per_src[src]['images'] += 1
        v_mean = None
        for g, j in zip(gts, judge(gts, preds, names)):
            c, x1, y1, x2, y2 = g
            gh, gw = y2 - y1, x2 - x1
            if j['cls'] in ('green-crossing', 'red-crossing'):
                if v_mean is None:
                    im = r.orig_img
                    v_mean = float(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 2].mean())
                lights.append(dict(id=len(lights) + 1, file=fn, source=src, cls=j['cls'], result=j['result'],
                                   pred=j['pred'], best_iou=j['best_iou'], h_px=round(gh, 1), w_px=round(gw, 1),
                                   h_over_w=round(gh / max(gw, 1e-6), 2), img_brightness=round(v_mean, 1),
                                   x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), tag=''))
            elif j['cls'] == 'crosswalk':
                near = gh >= 0.15 * H or y2 >= H - 2
                cws.append(dict(file=fn, source=src, result=j['result'], best_iou=j['best_iou'], near=near,h_frac=round(gh / H, 3)))
            per_src[src][f"{j['cls']}_n"] += 1
            per_src[src][f"{j['cls']}_{j['result']}"] += 1

    # ---------- summary
    def rate(rows, cond):
        rows = list(rows)
        return round(100 * sum(map(cond, rows)) / max(1, len(rows)), 2), len(rows)
    S = {}
    for cls in ('red-crossing', 'green-crossing'):
        L = [l for l in lights if l['cls'] == cls]
        S[cls] = dict(n=len(L), recall_pct=rate(L, lambda l: l['result'] == 'ok')[0],wrong_color_pct=rate(L, lambda l: l['result'] == 'wrong_color')[0],miss_pct=rate(L, lambda l: l['result'] == 'miss')[0])
    S['false_go_red_as_green_pct'] = S['red-crossing']['wrong_color_pct']
    M = [l for l in lights if l['result'] == 'miss']
    S['light_misses'] = len(M)
    S['misses_with_iou_0.3_0.5_pct'] = rate(M, lambda l: 0.3 <= l['best_iou'] < 0.5)[0]
    S['misses_no_overlap_pct'] = rate(M, lambda l: l['best_iou'] < 0.1)[0]
    S['misses_flat_gt_pct'] = rate(M, lambda l: l['h_over_w'] < 1.5)[0]
    bins = [(0, 20), (20, 40), (40, 80), (80, 1e9)]
    S['misses_by_height_px'] = {f'{lo}-{hi if hi < 1e9 else "inf"}': rate(M, lambda l, lo=lo, hi=hi: lo <= l['h_px'] < hi)[0] for lo, hi in bins}
    S['all_lights_by_height_px'] = {f'{lo}-{hi if hi < 1e9 else "inf"}': rate(lights, lambda l, lo=lo, hi=hi: lo <= l['h_px'] < hi)[0] for lo, hi in bins}
    S['recall_by_height_px'] = {f'{lo}-{hi if hi < 1e9 else "inf"}': rate([l for l in lights if lo <= l['h_px'] < hi], lambda l: l['result'] == 'ok') for lo, hi in bins}
    dark = [l for l in lights if l['img_brightness'] < 70]
    S['recall_dark_images(V<70)'] = rate(dark, lambda l: l['result'] == 'ok')
    S['crosswalk_recall'] = rate(cws, lambda c: c['result'] == 'ok')
    S['crosswalk_near_recall'] = rate([c for c in cws if c['near']], lambda c: c['result'] == 'ok')
    S['crosswalk_far_recall'] = rate([c for c in cws if not c['near']], lambda c: c['result'] == 'ok')
    S['note'] = 'rate values = [percent, count]. conf/IoU = %.2f/0.5' % a.conf
    (out / 'summary.json').write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding='utf-8')

    for fn_, rows in (('lights.csv', lights), ('crosswalk.csv', cws)):
        if rows:
            with open(out / fn_, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    srows = []
    for src, c in sorted(per_src.items()):
        row = dict(source=src, images=c['images'])
        for cls in ('crosswalk', 'green-crossing', 'red-crossing'):
            n = c[f'{cls}_n']; row[f'{cls}_n'] = n
            row[f'{cls}_recall_pct'] = round(100 * c[f'{cls}_ok'] / n, 1) if n else ''
        row['red_as_green'] = c['red-crossing_wrong_color']
        srows.append(row)
    with open(out / 'per_source.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0].keys())); w.writeheader(); w.writerows(srows)

    # ---------- missed-signal contact sheets
    T, per = 150, 48
    Ms = sorted(M, key=lambda l: l['id'])[:a.sheet_max]
    for s0 in range(0, len(Ms), per):
        tiles = []
        for l in Ms[s0:s0 + per]:
            im = cv2.imread(str(idir / l['file']))
            if im is None:
                continue
            Hh, Ww = im.shape[:2]
            bw, bh = l['x2'] - l['x1'], l['y2'] - l['y1']
            pad = int(max(bw, bh) * 1.5) + 10
            cx1, cy1 = max(0, l['x1'] - pad), max(0, l['y1'] - pad)
            cx2, cy2 = min(Ww, l['x2'] + pad), min(Hh, l['y2'] + pad)
            crop = im[cy1:cy2, cx1:cx2].copy()
            cv2.rectangle(crop, (l['x1'] - cx1, l['y1'] - cy1), (l['x2'] - cx1, l['y2'] - cy1), (0, 0, 255), 1)
            sc = T / max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))))
            tile = np.full((T + 28, T, 3), 255, np.uint8)
            tile[28:28 + crop.shape[0], :crop.shape[1]] = crop
            cv2.putText(tile, f"#{l['id']} {l['cls'][:5]} h{int(l['h_px'])}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
            cv2.putText(tile, f"iou{l['best_iou']} {l['source'][:12]}", (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1)
            tiles.append(tile)
        if not tiles:
            continue
        while len(tiles) % 8:
            tiles.append(np.full_like(tiles[0], 255))
        cv2.imwrite(str(out / f'misses_sheet_{s0 // per + 1}.jpg'), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
    print(json.dumps(S, ensure_ascii=False, indent=1))
    print('\nper source:')
    for r_ in srows:
        print(r_)
    print('\noutput:', out)


if __name__ == '__main__':
    main()
