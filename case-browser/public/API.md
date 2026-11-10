# Using the WSI archive API directly

No proxied API is provided by the case viewer app - call the archive yourself with your own Life Science AAI token. Every endpoint below is part of the archive's own EMPAIA WSI Service API (`$BASE/openapi.json` has the full spec); the app is just a thin browser UI over these same calls.

**Getting a token**: click "Copy access token" on the app's `/docs` page (linked from `/browse` as "Script docs & access token") - it's your current, valid Life Science AAI access token, good for ~1h. It works with any of the calls below, nothing app-specific about it.

```bash
TOKEN="<your access token>"
BASE="https://cbioportal.wsi-archive.bbmri-eric.eu/wsi-service/v3"
```

## List cases and slides

`/cases/` returns every case across every dataset the archive hosts, each with its `slides` array - there's no separate "list slides" endpoint, slide ids come from here. The app itself only shows cases in the `ratio` dataset (id starting with `ratio.`); the raw API doesn't filter, so do that client-side if you want the same scope.

The `ratio` dataset has two cohorts, told apart by the `org.site` prefix of the case and slide ids:

| Cohort | Prefix | Example slide id |
| --- | --- | --- |
| Real slides | `ratio.link` | `ratio.link.w.SLD_722c353d` |
| Synthetic slides | `ratio.slink` | `ratio.slink.w.SLD-0059afc9` |

Everything below works the same for both - just use a slide id (or `SITE`) from the cohort you want.

```bash
curl -H "Authorization: Bearer $TOKEN" "$BASE/cases/" | jq '[.[] | select(.id | startswith("ratio."))]'
```

```python
import requests

token = "<your access token>"
base = "https://cbioportal.wsi-archive.bbmri-eric.eu/wsi-service/v3"
headers = {"Authorization": f"Bearer {token}"}

cases = requests.get(f"{base}/cases/", headers=headers).json()
cases = [c for c in cases if c["id"].split(".")[0] == "ratio"]
print(f"{len(cases)} ratio cases")

slide_id = cases[0]["slides"][0]
print("first slide:", slide_id)
```

## Slide metadata

Dimensions, pyramid levels, tile size, format:

```bash
curl -H "Authorization: Bearer $TOKEN" "$BASE/slides/info?slide_id=ratio.link.w.SLD_722c353d"
```

## Thumbnails and individual tiles

Both need a Bearer header, so they can't be dropped straight into an `<img src>` in a browser.

```bash
# Thumbnail, longest side capped at 480px
curl -H "Authorization: Bearer $TOKEN" \
  "$BASE/slides/thumbnail/max_size/480/480?slide_id=ratio.link.w.SLD_722c353d" -o thumb.jpg

# A single pyramid tile: level 0 (highest resolution), tile column 0, row 0.
# Use slides/info above to find valid level/tile_x/tile_y ranges for a slide -
# each level's extent divided by the slide's tile_extent gives the tile grid.
curl -H "Authorization: Bearer $TOKEN" \
  "$BASE/slides/tile/level/0/tile/0/0?slide_id=ratio.link.w.SLD_722c353d" -o tile.jpg
```

```python
info = requests.get(f"{base}/slides/info", params={"slide_id": slide_id}, headers=headers).json()
print(info["format"], info["extent"], info["num_levels"], "levels")

tile = requests.get(
    f"{base}/slides/tile/level/0/tile/0/0",
    params={"slide_id": slide_id},
    headers=headers,
)
open("tile.jpg", "wb").write(tile.content)
```

## Download a single slide

Streams a zip of the slide's underlying storage files (`-f` makes curl fail on an HTTP error instead of saving the error body as `slide.zip`):

```bash
curl -fSL -H "Authorization: Bearer $TOKEN" "$BASE/slides/download?slide_id=ratio.link.w.SLD_722c353d" -o slide.zip
```

## Download a set of slides

There's no batch-download endpoint (`/slides/download` takes exactly one `slide_id`) - loop over the slide ids you want, same as the single-slide call:

```bash
# Replace SECOND_SLIDE_ID etc. with real ids from /cases/
for slide_id in ratio.link.w.SLD_722c353d SECOND_SLIDE_ID; do
  curl -fSL -H "Authorization: Bearer $TOKEN" \
    "$BASE/slides/download?slide_id=$slide_id" -o "$slide_id.zip"
done

# ...or every slide belonging to a given case:
curl -sH "Authorization: Bearer $TOKEN" "$BASE/cases/" \
  | jq -r '.[] | select(.id == "ratio.link.c.SLD_722c353d") | .slides[]' \
  | while read -r slide_id; do
      curl -fSL -H "Authorization: Bearer $TOKEN" \
        "$BASE/slides/download?slide_id=$slide_id" -o "$slide_id.zip"
    done
```

Slides are multi-GB, so stream them to disk rather than holding each one in memory. Writing to a `.part` file and renaming it only once complete means a finished `.zip` is never a truncated one:

```python
import concurrent.futures
import os

def download_slide(slide_id: str, out_dir: str = ".") -> None:
    out = os.path.join(out_dir, f"{slide_id}.zip")
    with requests.get(
        f"{base}/slides/download", params={"slide_id": slide_id}, headers=headers, stream=True
    ) as r:
        r.raise_for_status()
        with open(f"{out}.part", "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    os.replace(f"{out}.part", out)

case = next(c for c in cases if c["id"] == "ratio.link.c.SLD_722c353d")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(download_slide, case["slides"]))
```

## Download an entire site

A site (`org.site` prefix, i.e. a cohort: `ratio.link` or `ratio.slink`) can be ~2000 multi-GB slides - easily several TB. List every slide under the site, then loop the single-slide download, skipping files you already have so an interrupted run is safe to resume. Each slide is downloaded to a `.part` file and only renamed to `.zip` once complete, so a slide cut off mid-download is fetched again on the next run instead of being skipped as done.

```bash
SITE="ratio.link"   # or "ratio.slink" for the synthetic cohort

curl -sH "Authorization: Bearer $TOKEN" "$BASE/cases/" \
  | jq -r --arg site "$SITE" '.[] | select(.id | startswith($site + ".")) | .slides[]' \
  > slides.txt

mkdir -p "$SITE"
while read -r slide_id; do
  out="$SITE/$slide_id.zip"
  [ -s "$out" ] && { echo "skip $slide_id"; continue; }
  echo "downloading $slide_id..."
  curl -fSL -H "Authorization: Bearer $TOKEN" "$BASE/slides/download?slide_id=$slide_id" -o "$out.part" \
    && mv "$out.part" "$out"
done < slides.txt
```

```python
site = "ratio.link"  # or "ratio.slink" for the synthetic cohort
slide_ids = [s for c in cases if c["id"].startswith(f"{site}.") for s in c["slides"]]
print(f"{len(slide_ids)} slides under {site}")

os.makedirs(site, exist_ok=True)
for slide_id in slide_ids:
    if os.path.exists(f"{site}/{slide_id}.zip"):
        continue
    download_slide(slide_id, out_dir=site)  # streaming helper from the section above
```

Your token expires in ~1h - if downloads start failing partway through with 401s, copy a fresh one from the app and re-export `TOKEN`; already-downloaded files are skipped on the next run, and any leftover `.part` file is simply overwritten.
