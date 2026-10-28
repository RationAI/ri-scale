# xOpat 3 viewer in a musica pod

Rebuilds the viewer used to look at the slides, the predicted lymph nodes and the tissue classes (`scripts/make_overlays.py`,
`reports/*.py`). Everything is built from public sources inside the pod; nothing from patient data is stored here.

* **xOpat 3.1.0**, the release the cluster runs as `ghcr.io/rationai/xopat:3.1.0` (not the archived v2 line), configured like that
  deployment (`wsi_service` slide protocol, `RationaiStandaloneV3TileSource`), see `viewer.env.v3.json`.
* **WSI tile service**: `RationAI/wsi-service` at `8b4e529`, with its git submodules, path mapper over `~/viewer/data`
  (`MMCI/<slide>.mrxs`, `MUG/<slide>.svs`). Login is disabled, so both services listen on the pod's loopback only.
* **`overlay-url` plugin** (ours, no xOpat file is edited): `?overlay=<url>` (repeatable) and `?overlay_opt=<url>` (a missing file is
  skipped) import GeoJSON in level-0 pixels as annotations once the slide is open; `?stroke=<w>` sets the outline width. Features get the
  preset `lymph_node` or `tissue_<class>` (see `staticPresets` in the config).
* Overlay and report files are served from `xopat3/overlay-data/` (a registered `core.server.staticRoots`; the server refuses symlinks that
  leave the app folder).

```
bash viewer/setup_viewer.sh          # once, inside the pod
bash ~/viewer/start.sh               # starts the tile service (8080) and xOpat (9000)
# from the workstation (node and port: ~/pod/NODE, ~/pod/PORT):
ssh -N -J musica -i ~/.ssh/id_musica -p <PORT> -L 127.0.0.1:9000:127.0.0.1:9000 -L 127.0.0.1:8080:127.0.0.1:8080 jp50463@<NODE>
http://127.0.0.1:9000/?slides=MMCI/<slide>.mrxs&overlay=/overlay-data/overlays/<slide>.json
```

## Image overlays (preferred over the GeoJSON overlays)

`scripts/make_overlay_images.py` writes one RGBA PNG per slide that covers the slide's extent: `<slide>.tissue.png` (tissue classes, 4 x 4 pixels
per 128 um tile) and `<slide>.nodes.png` (predicted lymph nodes). Put them into `xopat3/overlay-data/rasters/`. xOpat shows them as layers of a
session (slide as background, each image a visualization shader of type `identity`, protocol `plain_image`, `imageSmoothingEnabled: false`);
`reports/*.py --rasters-dir` builds such `?visualization=<json>` links. Each layer has its own visibility and opacity in the viewer.
The GeoJSON route (`overlay-url` plugin, `scripts/make_overlays.py`) still works but is no longer used by the reports.
