"""Do the OWLv2 'spot' regions land on the mite? Uses the mite boxes in gt.csv (bee domain only, DATASET=bees).

For every Varroa photo with a mite box: does any spot region's centre fall inside a mite box? For healthy photos:
how many spot regions are found anyway (false alarms)? Output: results/region_check.csv
"""
import ast

import pandas as pd

from common import DATASET, REGIONS, RESULTS, load_meta, read_json

assert DATASET == "bees"
meta = load_meta()
reg = read_json(REGIONS / "regions.json")
rows = []
for p, lab, boxes in zip(meta.path, meta.label, meta.mite_boxes):
    r = reg.get(p)
    if r is None:
        continue
    spots = [x for x in r["regions"] if x["kind"] == "spot"]
    boxes = ast.literal_eval(boxes)
    hit = any(b[0] <= (s["box"][0] + s["box"][2]) / 2 <= b[2] and b[1] <= (s["box"][1] + s["box"][3]) / 2 <= b[3]
              for s in spots for b in boxes)
    rows.append(dict(path=p, label=lab, n_spots=len(spots), has_box=bool(boxes), spot_on_mite=hit,
                     object_found=r["object"] is not None))
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "region_check.csv", index=False)
inf = df[(df.label == 1) & df.has_box]
print(f"Varroa photos with a mite box: {len(inf)}; a spot region on the mite in {inf.spot_on_mite.mean():.1%}")
print(f"mean spot regions per photo: varroa {df[df.label == 1].n_spots.mean():.2f}, healthy {df[df.label == 0].n_spots.mean():.2f}")
print(f"object (bee) box found: {df.object_found.mean():.1%}")
