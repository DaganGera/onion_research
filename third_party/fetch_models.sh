# Resumable downloads (curl -C -) of the models whose HF-hub downloads kept stalling on this network.
set -u
get() {  # repo  file  outdir
  mkdir -p "$3"
  for i in $(seq 1 60); do
    curl -sfL -C - --speed-limit 20000 --speed-time 60 -o "$3/$2" "https://huggingface.co/$1/resolve/main/$2" && return 0
    sleep 3
  done
  echo "FAILED $1/$2"; exit 1
}
M=third_party/models
for f in config.json merges.txt vocab.json tokenizer.json tokenizer_config.json; do get FacebookAI/roberta-base $f $M/roberta-base; done
touch $M/roberta-base/.complete; echo ROBERTA_OK
for f in config.json model.safetensors preprocessor_config.json; do get facebook/dinov2-small $f $M/dinov2-small; done
touch $M/dinov2-small/.complete; echo DINO_OK
for f in chat_template.json config.json generation_config.json merges.txt preprocessor_config.json tokenizer.json \
         tokenizer_config.json video_preprocessor_config.json vocab.json model.safetensors; do
  get Qwen/Qwen3-VL-2B-Instruct $f $M/Qwen3-VL-2B-Instruct
done
touch $M/Qwen3-VL-2B-Instruct/.complete; echo QWEN_OK
