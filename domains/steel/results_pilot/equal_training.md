Macro-F1 on the test set (mean ± std over 3 seeds), every method re-run on the same GPU.
last-epoch = usual training; best-epoch = 2x epochs, best of 10 validation checkpoints.

|                                                    | 4           | 16          |
|:---------------------------------------------------|:------------|:------------|
| ('CLIP-Adapter', 'best-epoch')                     | 0.911 ± nan | 0.987 ± nan |
| ('CLIP-Adapter [clip_l14]', 'best-epoch')          | 0.924 ± nan | 0.983 ± nan |
| ('GraphAdapter', 'best-epoch')                     | 0.324 ± nan | 0.966 ± nan |
| ('GraphAdapter [clip_l14]', 'best-epoch')          | 0.823 ± nan | 0.963 ± nan |
| ('PRGA', 'as-is')                                  | 0.922 ± nan | 0.994 ± nan |
| ('PRGA + DINOv2 cache', 'as-is')                   | 0.970 ± nan | 0.994 ± nan |
| ('PRGA + DINOv2 cache [clip_l14]', 'as-is')        | 0.964 ± nan | 0.992 ± nan |
| ('PRGA [clip_l14]', 'as-is')                       | 0.928 ± nan | 0.987 ± nan |
| ('PlantCaFo-style cache', 'best-epoch')            | 0.964 ± nan | 0.990 ± nan |
| ('PlantCaFo-style cache [clip_l14]', 'best-epoch') | 0.961 ± nan | 0.981 ± nan |
| ('TaskRes', 'best-epoch')                          | 0.907 ± nan | 0.984 ± nan |
| ('TaskRes [clip_l14]', 'best-epoch')               | 0.927 ± nan | 0.982 ± nan |
| ('Tip-Adapter-F', 'best-epoch')                    | 0.891 ± nan | 0.986 ± nan |
| ('Tip-Adapter-F [clip_l14]', 'best-epoch')         | 0.928 ± nan | 0.978 ± nan |
