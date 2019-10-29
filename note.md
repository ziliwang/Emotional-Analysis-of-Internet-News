### model performance
pretrained-lm | file-name | online |local cv 5 fold | preprocess | best_stack
---|---|---|---|---|---
robert_l24 | submit_20191019_1023.csv | 0.81028026000 | 0.804879 |fixed | Y
|robert_l12 | submit_20191017_1601.csv | 0.80695093000 | 0.806693 | fixed |
wwm_bert  |  submit_20191018_2228.csv | 0.81096435000 | 0.810670 | fixed  |  Y
xlnet_l12 |  submit_20191023_1026.csv | 0.80419731000 | 0.808558 | dynamic | Y
robert_l24 | submit_20191016_2232.csv | 0.80771929000 | 0.807993 | dynamic|
xlnet_l12 |  submit_20191024_1020.csv | 0.80685902000 | 0.809733 | fixed |

### useful skill
1. learn rate decay across the layers improved the performance in bert-l12. but damaged in bert-l24 or xlnet.
2. dropout rate set to 0.3 improve the performance in bert-l12.


### logs
