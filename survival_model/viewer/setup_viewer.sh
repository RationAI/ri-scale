#!/usr/bin/env bash
# Build the xOpat 3.1.0 viewer + the WSI tile service from the public repositories inside a pod (home is shared between pods).
# Run inside the pod:  bash setup_viewer.sh   then   bash ~/viewer/start.sh
# The viewer is reached through an ssh tunnel only (ports 9000 and 8080 on the pod's loopback), see README.md.
set -euo pipefail
V=$HOME/viewer; HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p $V; cd $V
echo "== node 22"
[ -x $V/node/bin/node ] || { curl -fsSL https://nodejs.org/dist/v22.12.0/node-v22.12.0-linux-x64.tar.xz | tar -xJ -C $V && mv $V/node-v22.12.0-linux-x64 $V/node; }
export PATH=$V/node/bin:$PATH
echo "== xopat 3.1.0 (the release deployed as ghcr.io/rationai/xopat:3.1.0)"
[ -d xopat3/.git ] || git clone -q --branch 3.1.0 --depth 1 https://github.com/RationAI/xopat.git xopat3
(cd xopat3 && npm install --no-audit --no-fund && npm run minify)
echo "== wsi-service (RationAI fork, with its submodules)"
[ -d wsi-service/.git ] || git clone -q https://github.com/RationAI/wsi-service.git wsi-service
(cd wsi-service && git checkout -q 8b4e529 && git submodule update --init --recursive)
# the container's Python has no ensurepip: create the environment without pip and bootstrap it
[ -x wsi-env/bin/pip ] || { rm -rf wsi-env; python3 -m venv --without-pip wsi-env; curl -fsSL https://bootstrap.pypa.io/get-pip.py | ./wsi-env/bin/python - -q; }
./wsi-env/bin/pip install -q ./wsi-service ./wsi-service/wsi_service_base_plugins/{openslide,pil,tifffile,tiffslide} openslide-bin
echo "== data folder, config, plugin, launcher"
mkdir -p $V/data xopat3/overlay-data/overlays xopat3/overlay-data/reports xopat3/plugins/overlay-url
ln -sfn /data/fs201316/rh02261/data/mmci_lymph_nodes $V/data/MMCI
ln -sfn /data/fs201316/rh02261/data/mug_lymph_nodes  $V/data/MUG
cp $HERE/viewer.env.v3.json $V/viewer.env.v3.json; cp $HERE/start.sh $V/start.sh; chmod +x $V/start.sh
cp $HERE/overlay-url/include.json $HERE/overlay-url/overlay-url.js xopat3/plugins/overlay-url/
echo "== done"
