set -u
D=third_party/models/bioclip; mkdir -p $D
for f in open_clip_config.json open_clip_pytorch_model.bin; do
  for i in $(seq 1 200); do
    curl -sfL -C - --speed-limit 5000 --speed-time 60 -o "$D/$f" "https://huggingface.co/imageomics/bioclip/resolve/main/$f" && break
    sleep 3
  done
done
[ "$(stat -c %s $D/open_clip_pytorch_model.bin)" = "598599013" ] && touch $D/.complete && echo BIOCLIP_OK || echo BIOCLIP_FAILED
