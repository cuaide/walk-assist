# Labeling rules

These rules define the v1 dataset (`ds_v4`). Three classes; there is no separate class for countdown timers.

| ID | Class | What is labeled |
| ---: | --- | --- |
| 0 | `crosswalk` | Zebra crosswalk painted on the road |
| 1 | `green-crossing` | Pedestrian signal head with the green walking-person lamp lit |
| 2 | `red-crossing` | Pedestrian signal head with the red standing-person lamp lit |

## Pedestrian signals (`green-crossing`, `red-crossing`)

1. **Find the pedestrian signal first.** Vehicle traffic lights (round or arrow lamps on horizontal arms) are never labeled.
2. **The lit lamp sets the class.** Red standing person lit → `red-crossing`; green walking person lit → `green-crossing`.
3. **Box the whole signal head, not only the lit cell.** One box covers both lamp cells (top red, bottom green) plus a countdown timer attached directly to the head, whether the timer is lit or not. The larger, fixed-shape box keeps the signal detectable from far away, and it does not change size with the lamp color, so the model cannot use box shape as a color cue.
4. **Exclude** poles, brackets, and timers mounted separately (on the other side of the pole or at another height).
5. **Leave unlabeled** signals that are off, facing away, or whose lamp color cannot be seen.

## Crosswalks (`crosswalk`)

1. **One tight box per crosswalk** around its white stripes. If the stripes run to the image border, that side of the box touches the border.
2. **At least three consecutive white stripes** must be distinguishable; otherwise (e.g. far away) it is not labeled. Faded or cracked stripes count if they can be told apart.
3. **Occlusion:** a crosswalk partly hidden by cars, people or poles still gets one box that spans the occluder.
4. **Separate boxes** for crosswalks in different directions (e.g. at an L-shaped intersection) and for crosswalks split by a median or traffic island.
5. **Not labeled:** lane markings, stop lines, curbs, sidewalk tiles. Background inside the box of a diagonal crosswalk is fine.

## Known deviations in v1

Some timer-integrated heads in the PTL sources are still boxed around the lit cell only, and some visible signals for the perpendicular crossing are unlabeled. They were found after the test run and are left unchanged in v1; they are scheduled for the next relabel.
