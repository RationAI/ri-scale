#!/usr/bin/env bash
# Start the WSI tile service (127.0.0.1:8080, login disabled, localhost origins only) and xOpat (127.0.0.1:9000). Reach them through the ssh tunnel only.
V=$HOME/viewer; export PATH=$V/node/bin:$PATH
if ! curl -s -m 3 localhost:8080/alive >/dev/null; then
  ( export WS_DATA_DIR=$V/data WS_LOCAL_MODE=wsi_service.paths_mapper:PathsMapper WS_CORS_ALLOW_CREDENTIALS=False WS_DEBUG=False
    export WS_CORS_ALLOW_ORIGINS='["http://localhost:9000","http://127.0.0.1:9000"]' WS_ENABLE_VIEWER_ROUTES=True WS_ENABLE_LOCAL_ROUTES=True WS_DISABLE_OPENAPI=False
    export WS_MAPPER_ADDRESS="http://localhost:8080/slides/storage?slide={slide_id}" WS_IMAGE_HANDLE_CACHE_SIZE=50 WS_INACTIVE_HISTO_IMAGE_TIMEOUT_SECONDS=600
    export WS_MAX_RETURNED_REGION_SIZE=25000000 WS_MAX_THUMBNAIL_SIZE=512 WS_API_V3_INTEGRATION=wsi_service.api.v3.integrations.disable_auth:DisableAuth
    cd $V/wsi-service && setsid nohup $V/wsi-env/bin/python -m uvicorn wsi_service.app:app --host 127.0.0.1 --port 8080 > $V/wsi-service.log 2>&1 & )
fi
if ! curl -s -m 3 -o /dev/null localhost:9000/; then
  ( export XOPAT_NODE_HOST=127.0.0.1 XOPAT_NODE_PORT=9000 XOPAT_ENV=$V/viewer.env.v3.json
    cd $V/xopat3 && setsid nohup node index.js > $V/xopat.log 2>&1 & )
fi
sleep 10
curl -s -m 3 localhost:8080/alive >/dev/null && echo "wsi-service up" || echo "wsi-service NOT up (see ~/viewer/wsi-service.log)"
curl -s -m 3 -o /dev/null localhost:9000/ && echo "xopat up" || echo "xopat NOT up (see ~/viewer/xopat.log)"
