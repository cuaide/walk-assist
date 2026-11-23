import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASSES = ['crosswalk', 'traffic_light', 'text_region']
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def index_images(root):
    by_name, by_stem = {}, {}
    for p in Path(root).rglob('*'):
        if p.suffix.lower() in IMG_EXT and not any(part.lower() == 'mask' for part in p.parts):
            by_name.setdefault(p.name, p)
            by_stem.setdefault(p.stem, p)
    return by_name, by_stem


class Writer:

    def __init__(self, out, prefix, link=False):
        self.img_dir = Path(out) / 'images'
        self.lbl_dir = Path(out) / 'labels'
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        self.prefix, self.link = prefix, link
        self.n_img, self.counts = 0, Counter()

    def write(self, img_path, w, h, boxes, keep_empty=False):
        lines = []
        for c, x1, y1, x2, y2 in boxes:
            x1, x2 = sorted((max(0.0, min(w, x1)), max(0.0, min(w, x2))))
            y1, y2 = sorted((max(0.0, min(h, y1)), max(0.0, min(h, y2))))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f'{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} '
                         f'{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}')
            self.counts[CLASSES[c]] += 1
        if not lines and not keep_empty:
            return False
        name = f'{self.prefix}_{Path(img_path).stem}'
        dst = self.img_dir / (name + Path(img_path).suffix.lower())
        if not dst.exists():
            if self.link:
                dst.symlink_to(Path(img_path).resolve())
            else:
                shutil.copy2(img_path, dst)
        (self.lbl_dir / f'{name}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
        self.n_img += 1
        return True

    def report(self, what):
        print(f'[{what}] converted {self.n_img} images, boxes: {dict(self.counts)}')
        if self.n_img == 0:
            print('  Nothing was converted. Check the label structure with `inspect`.', file=sys.stderr)


def image_size(path):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    return (img.shape[1], img.shape[0]) if img is not None else (None, None)


def poly_box(pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def is_crosswalk(*texts):
    return any(t and 'crosswalk' in t.lower() for t in texts)


def cvat_images(root):
    for xml_path in Path(root).rglob('*.xml'):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as e:
            print(f'  failed to read XML {xml_path}: {e}', file=sys.stderr)
            continue
        for img in tree.getroot().iter('image'):
            yield img


def aihub_bbox(src, out, link, keep_classes=('traffic_light',), **_):
    by_name, _ = index_images(src)
    wr = Writer(out, 'aihubbox', link)
    labels_seen = Counter()
    for img in cvat_images(src):
        p = by_name.get(Path(img.get('name', '')).name)
        if p is None:
            continue
        w, h = float(img.get('width')), float(img.get('height'))
        boxes = []
        for b in img.iter('box'):
            lab = b.get('label', '')
            labels_seen[lab] += 1
            if lab == 'traffic_light' and 'traffic_light' in keep_classes:
                c = 1
            elif lab == 'traffic_sign' and 'traffic_sign' in keep_classes:
                c = 2
            else:
                continue
            boxes.append((c, float(b.get('xtl')), float(b.get('ytl')), float(b.get('xbr')), float(b.get('ybr'))))

        wr.write(p, w, h, boxes, keep_empty=random.random() < 0.1)
    print(f'  top 10 labels seen in XML: {labels_seen.most_common(10)}')
    wr.report('AI Hub bounding boxes')


def aihub_surface(src, out, link, **_):
    by_name, _ = index_images(src)
    wr = Writer(out, 'aihubsurf', link)
    seen = Counter()
    for img in cvat_images(src):
        p = by_name.get(Path(img.get('name', '')).name)
        if p is None:
            continue
        w, h = float(img.get('width')), float(img.get('height'))
        boxes = []
        for poly in img.iter('polygon'):
            attrs = [a.text or '' for a in poly.iter('attribute')] + [a.get('value', '') for a in poly.iter('attribute')]
            seen[f"{poly.get('label')}/{'|'.join(t for t in attrs if t)}"] += 1
            if not is_crosswalk(poly.get('label'), *attrs):
                continue
            pts = [tuple(map(float, xy.split(','))) for xy in poly.get('points', '').split(';') if xy]
            if len(pts) >= 3:
                boxes.append((0, *poly_box(pts)))
        wr.write(p, w, h, boxes, keep_empty=random.random() < 0.1)
    print(f'  top 15 polygon labels/attributes: {seen.most_common(15)}')
    wr.report('AI Hub surface masking')


# ---------------------------------------------------------------- AI Hub outdoor Korean text (JSON)
def find_boxes_json(obj):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if kl in ('bbox', 'box', 'bounding_box') and isinstance(v, list):
                flat = np.asarray(v, dtype=float).reshape(-1)
                if flat.size == 4:
                    x, y, a, b = flat
                    # [x,y,w,h] or [x1,y1,x2,y2]? if the 3rd/4th values are smaller than the 1st/2nd, treat them as w,h
                    out.append((x, y, x + a, y + b) if (a < x or b < y) or kl == 'bbox' else (x, y, a, b))
                elif flat.size >= 8:
                    out.append(poly_box(flat))
            elif kl in ('bbox', 'box') and isinstance(v, dict) and {'x', 'y'} <= set(v):
                x, y = float(v['x']), float(v['y'])
                ww, hh = float(v.get('width', v.get('w', 0))), float(v.get('height', v.get('h', 0)))
                out.append((x, y, x + ww, y + hh))
            else:
                out += find_boxes_json(v)
    elif isinstance(obj, list):
        for v in obj:
            out += find_boxes_json(v)
    return out


TEXT_KEYS = ('text', 'transcription', 'value', 'word', 'char', 'label')


def find_text_items(obj):
    out = []
    if isinstance(obj, dict):
        if any(k.lower() in ('bbox', 'box', 'bounding_box') for k in obj):
            txt = next((str(v) for k, v in obj.items() if k.lower() in TEXT_KEYS and isinstance(v, (str, int))), None)
            bx = find_boxes_json({k: v for k, v in obj.items() if k.lower() in ('bbox', 'box', 'bounding_box')})
            if txt is not None and bx:
                out.append((bx[0], txt))
        for v in obj.values():
            if isinstance(v, (dict, list)):
                out += find_text_items(v)
    elif isinstance(obj, list):
        for v in obj:
            out += find_text_items(v)
    return out


def aihub_text(src, out, link, limit=5000, **_):
    by_name, by_stem = index_images(src)
    wr = Writer(out, 'aihubtext', link)
    jsons = sorted(Path(src).rglob('*.json'))
    random.shuffle(jsons)
    gt, n_txt = open(Path(out) / 'ocr_gt.jsonl', 'w', encoding='utf-8'), 0
    for jp in jsons:
        if wr.n_img >= limit:
            break
        try:
            data = json.loads(jp.read_text(encoding='utf-8-sig'))
        except Exception as e:
            print(f'  failed to read JSON {jp}: {e}', file=sys.stderr)
            continue
        p = by_stem.get(jp.stem)
        if p is None:
            fn = json.dumps(data, ensure_ascii=False)
            for name in by_name:
                if name in fn:
                    p = by_name[name]
                    break
        if p is None:
            continue
        w, h = image_size(p)
        if w is None:
            continue
        boxes = drop_containers(find_boxes_json(data))
        if wr.write(p, w, h, [(2, *b) for b in boxes]):
            items = find_text_items(data)
            gt.write(json.dumps({'image': f'aihubtext_{Path(p).stem}{Path(p).suffix.lower()}', 'width': w, 'height': h,'items': [{'box': [round(float(c), 1) for c in bx], 'text': t} for bx, t in items]},ensure_ascii=False) + '\n')
            n_txt += len(items)
    gt.close()
    wr.report('AI Hub outdoor Korean text')
    print(f'  OCR ground truth: {Path(out) / "ocr_gt.jsonl"} ({n_txt} box+text items)'
          + ('' if n_txt else '  <- no text found. Check the JSON text key name with `inspect`.'))
    print('  If large signboard boxes remain: python refine_text.py <this output folder> <new folder> --preview')


def drop_containers(boxes, n_inside=2):
    def inside(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy >= 0.9 * max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    return [b for b in boxes
            if sum(1 for o in boxes if o is not b and o != b and inside(o, b)) < n_inside]


def inspect(src, **_):
    shown = 0
    for p in sorted(Path(src).rglob('*')):
        if p.suffix.lower() == '.xml' and shown < 2:
            root = ET.parse(p).getroot()
            print(f'== {p}\n  root <{root.tag}>, child tags: {Counter(e.tag for e in root.iter()).most_common(12)}')
            for e in list(root.iter())[:15]:
                print(f'   <{e.tag} {dict(list(e.attrib.items())[:6])}> {(e.text or "").strip()[:40]}')
            shown += 1
        elif p.suffix.lower() == '.json' and shown < 2:
            txt = p.read_text(encoding='utf-8-sig', errors='replace')
            print(f'== {p}\n{txt[:1500]}')
            shown += 1
        elif p.suffix.lower() == '.csv' and shown < 2:
            print(f'== {p}\n' + ''.join(p.open(encoding='utf-8', errors='replace').readlines()[:5]))
            shown += 1
    n_img = sum(1 for p in Path(src).rglob('*') if p.suffix.lower() in IMG_EXT)
    print(f'{n_img} images')


def make_yaml(dirs, val_frac=0.1, out='data.yaml', **_):
    random.seed(0)
    train, val, test = [], [], []
    for d in dirs:
        imgs = sorted(str(p.resolve()) for p in (Path(d) / 'images').glob('*'))
        if 'test' in Path(d).name:
            test += imgs
            continue
        random.shuffle(imgs)
        k = int(len(imgs) * val_frac)
        val += imgs[:k]
        train += imgs[k:]
    base = Path(out).resolve().parent
    for name, lst in (('train', train), ('val', val), ('test', test)):
        (base / f'{name}.txt').write_text('\n'.join(lst) + '\n')
    yaml = [f'path: {base.as_posix()}', 'train: train.txt', 'val: val.txt']
    if test:
        yaml.append('test: test.txt')
    yaml += ['names:'] + [f'  {i}: {c}' for i, c in enumerate(CLASSES)]
    Path(out).write_text('\n'.join(yaml) + '\n')
    print(f'{out}: train {len(train)}, val {len(val)}, test {len(test)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', choices=['inspect', 'aihub-bbox', 'aihub-surface', 'aihub-text', 'yaml'])
    ap.add_argument('src', nargs='+', help='source folder (for yaml: several conversion output folders)')
    ap.add_argument('--out', help='output folder (can also be given as the second positional argument after src)')
    ap.add_argument('--link', action='store_true', help='symlink images instead of copying (Linux/macOS)')
    ap.add_argument('--limit', type=int, default=5000)
    ap.add_argument('--with-sign', action='store_true', help='add AI Hub traffic_sign as text_region (2)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    if a.kind == 'yaml':
        return make_yaml(a.src, out=a.out or 'data.yaml')
    src = a.src[0]
    out = a.out or (a.src[1] if len(a.src) > 1 else None)
    if a.kind == 'inspect':
        return inspect(src)
    if out is None:
        ap.error('an output folder is required')
    fn = {'aihub-bbox': aihub_bbox, 'aihub-surface': aihub_surface, 'aihub-text': aihub_text}[a.kind]
    keep = ('traffic_light', 'traffic_sign') if a.with_sign else ('traffic_light',)
    fn(src, out, a.link, limit=a.limit, keep_classes=keep)


if __name__ == '__main__':
    main()
