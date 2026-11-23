import argparse, os, yaml, pandas as pd
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', required=True, help='re-split dataset with valid/, data.yaml and groups.csv')
ap.add_argument('--weights', required=True)
ap.add_argument('--out', default='valid_by_origin', help='work folder for subset links and val runs')
ap.add_argument('--csv', default='valid_by_origin.csv')
ap.add_argument('--imgsz', type=int, default=640)
a = ap.parse_args()
DS, W, TMP = a.data_root, a.weights, os.path.abspath(a.out)

g = pd.read_csv(f'{DS}/groups.csv', encoding='utf-8-sig')
g = g[g.split == 'valid']
g['origin'] = g.old_split.map(lambda s: 'old_train' if s == 'train' else 'old_valid_test')
names = yaml.safe_load(open(f'{DS}/data.yaml'))['names']
print(g.origin.value_counts().to_dict())

m = YOLO(W)
rows = []
for origin, sub in g.groupby('origin'):
    d = f'{TMP}/{origin}'
    os.makedirs(f'{d}/images', exist_ok=True); os.makedirs(f'{d}/labels', exist_ok=True)
    for f in sub.file:
        st = os.path.splitext(f)[0]
        for a, b in [(f'{DS}/valid/images/{f}', f'{d}/images/{f}'), (f'{DS}/valid/labels/{st}.txt', f'{d}/labels/{st}.txt')]:
            if os.path.exists(a) and not os.path.exists(b): os.symlink(a, b)
    y = f'{d}/data.yaml'
    yaml.safe_dump(dict(path=d, train='images', val='images', nc=len(names), names=names), open(y, 'w'))
    r = m.val(data=y, imgsz=a.imgsz, batch=16, verbose=False, plots=False, project=TMP, name=f'val_{origin}', exist_ok=True)
    for k, ci in enumerate(r.box.ap_class_index):            # only classes present in this subset
        p, rc, ap50, ap = r.box.class_result(k)
        rows.append(dict(origin=origin, n_img=len(sub), cls=names[int(ci)], P=p, R=rc, AP50=ap50, AP50_95=ap))
df = pd.DataFrame(rows)
print(df.pivot(index='cls', columns='origin', values=['AP50', 'AP50_95', 'R']).round(3).to_string())
df.to_csv(a.csv, index=False)
print('saved:', a.csv)
