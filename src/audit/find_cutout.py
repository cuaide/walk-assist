import argparse, os, glob, numpy as np, pandas as pd, cv2
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', required=True, help='dataset with train/valid/test/images')
ap.add_argument('--out', default='audit', help='folder for cutout_<split>.csv and cutout_sheet.jpg')
a = ap.parse_args()
DS, OUT_DIR = a.data_root, a.out
SPLITS  = ['valid', 'test', 'train']        # train only for reference (cutout there is fine)
os.makedirs(OUT_DIR, exist_ok=True)

def squares(path):
    im = cv2.imread(path)
    if im is None: return path, []
    h, w = im.shape[:2]; s = min(h, w)
    black = (im.max(2) <= 10).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(black, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if not (0.02 * s <= bw <= 0.25 * s and 0.02 * s <= bh <= 0.25 * s): continue
        if not (0.8 <= bw / bh <= 1.25): continue          # roughly square
        if area < 0.92 * bw * bh: continue                  # solid fill (not a shadow / dark object)
        out.append((int(x), int(y), int(bw), int(bh)))
    return path, out

hits_all = []
for sp in SPLITS:
    files = sorted(glob.glob(f'{DS}/{sp}/images/*'))
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(squares, files))
    hits = [(os.path.basename(p), len(q), q) for p, q in res if q]
    pd.DataFrame(hits, columns=['image', 'n_squares', 'boxes']).to_csv(f'{OUT_DIR}/cutout_{sp}.csv', index=False)
    print(f'{sp:6} {len(hits):5d} / {len(files)} images with cutout squares ({100*len(hits)/max(1,len(files)):.1f}%)')
    if sp != 'train': hits_all += [(sp, p, q) for p, q in res if q]

tiles = []
for sp, p, q in hits_all[:24]:
    im = cv2.imread(p)
    for x, y, bw, bh in q: cv2.rectangle(im, (x, y), (x + bw, y + bh), (0, 0, 255), max(2, im.shape[0] // 200))
    t = cv2.resize(im, (256, 256)); cv2.putText(t, f'{sp} {os.path.basename(p)[:22]}', (4, 16), 0, 0.4, (0, 255, 255), 1)
    tiles.append(t)
if tiles:
    while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(f'{OUT_DIR}/cutout_sheet.jpg', np.vstack([np.hstack(tiles[i:i+6]) for i in range(0, len(tiles), 6)]))
    print('Sheet:', f'{OUT_DIR}/cutout_sheet.jpg')
