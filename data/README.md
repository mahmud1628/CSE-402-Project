# Local datasets

The project keeps Stage 1 benchmarks and Stage 2 football data in this directory:

```text
data/
  *.txt.gz                         # Stage 1 SNAP benchmark archives
  competitions.json                # StatsBomb competition/season metadata
  matches/<competition>/<season>.json
  events/<match_id>.json
  lineups/<match_id>.json
```

The football files were moved here from the workspace's former
`open-data/data/` directory. Stage 2 defaults to this location; no `--data`
argument is needed. Stage 1 continues to use its existing benchmark archives.

Raw datasets are excluded from new Git additions. A fresh clone needs the
selected football data supplied locally before Stage 2 will run:

1. Read `experiments/stage2/selected_matches.json` for the selected match IDs and
   competition/season pairs.
2. Obtain the corresponding `events/<match_id>.json` and
   `lineups/<match_id>.json` files from
   [StatsBomb Open Data](https://github.com/hudl/open-data).
3. Preserve the competition/season layout for the matching metadata files and
   include `competitions.json`.

The manifest's `event_path` and `metadata_path` fields record paths relative to
the original workspace (the parent of `CSE-402-Project/`). The runner uses match
IDs and its configured data root, so those provenance fields do not constrain
where a new checkout can be located.

The runner never downloads missing football files. It reports missing files
instead. Only the selected 50 event/lineup pairs are needed, even when the local
directory contains additional matches.

Keep [the upstream attribution notes](STATSBOMB_README.md) and
[license](STATSBOMB_LICENSE.pdf) with the data. Existing tracked SNAP archives
remain tracked until explicitly removed from Git's index.
