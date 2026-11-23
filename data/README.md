# Data

This folder contains **no images and no labels**. It records how the v1 dataset (`ds_v4`) was split, so that the evaluation can be audited without redistributing data.

| File | Contents |
| --- | --- |
| `split_manifest.csv` | One row per image: `file`, `group`, `split` (train/valid/test), `source` family. 8,291 rows (6,247 / 1,163 / 881). |
| `subsets/test_{full,no_seq,no_aug,clean,korea}.txt` | Test image lists for the five subsets, fixed before the test run (881 / 841 / 823 / 789 / 492). |
| `data.yaml` | Template for Ultralytics. Set `path` to your local copy. |

```yaml
path: /path/to/ds_v4
train: train/images
val: valid/images
test: test/images
nc: 3
names: [crosswalk, green-crossing, red-crossing]
```

## Sources

| Source | Where to get it | Licence / terms | Labels in this project |
| --- | --- | --- | --- |
| AI Hub pedestrian data (`MP_SEL_`) and stereo data (`ZED*`), Korea | [aihub.or.kr](https://aihub.or.kr) (approval required) | AI Hub terms; images and derived labels are not redistributed | Re-annotated with the v1 rules |
| PTL dataset (`heon_`, `john_`, `sam_`) | [ImVisible](https://github.com/samuelyu2002/ImVisible) | See the repository | Re-annotated with the v1 rules |
| Roboflow Universe: *Pedestrian Traffic Lights Final*, *Crosswalk&pedestrian light* | [link](https://universe.roboflow.com/test-xi95g/pedestrian-traffic-lights-final), [link](https://universe.roboflow.com/cap-8nhra/crosswalk-pedestrian-light) | CC BY 4.0 | Re-annotated with the v1 rules |
| Mixed web images (`gpl-`, `Cross`, `Strisce`, numeric names) | Via the Roboflow collections above | Original attribution partly unresolved | Re-annotated with the v1 rules |
| Own phone photos (`IMG_`, own) | Not released (bystanders visible) | — | Annotated by the author |

The watermarked stock-video preview `rpl` was removed from all splits before the test run and is not part of `ds_v4`.

## Rebuilding ds_v4

1. Collect the images above and annotate them with the rules in the main README (3 classes; signal box = whole head plus attached countdown timer).
2. Either place each file into the split given by `split_manifest.csv` (exact reproduction), or regroup from scratch:
   `python src/data/resplit_groups.py <export_dir> --out ds_v3 --gap 30 --stratify-source`, then delete `rpl*` from all splits.
3. Check the split: `src/audit/leak_followup.py` (filename sequences and near-duplicates across splits) and `src/audit/find_cutout.py` (offline-augmented copies).

Images you cannot obtain (for example own photos) are simply missing; report results on the remaining files and state the difference.
