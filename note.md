baseline: best score 0.8022, best epoch 2, best loss 0.2942, last score 0.7919(0.1860)
m2_v1: add batchnorm, best score 0.8014, best epoch 2, best loss 0.3897, last 0.7904(0.2840) -> ok
m3_v1: add maxpool(shared pool weight), 2e-5 + 3 epcoh, best(0.7898/3/0.2610)
m3_v1: 2e-5 + 5 epoch, best(0.7905/4/0.1518), last(0.7797/0.0981)
m4_v1: add maxpool, 2e-5 + 3 epoch, best(0.7943/3/0.2035) -> ok
m4_v1: 2e-5 + 3 epoch, best(0.7906/3/0.2430)
m1_v2: only head, 2e-5 + 3 epoch, best(0.7855/3/0.1867)
m1_v3: ignore part tail, 2e-5 + 3 epch, best(0.7908/2/0.2931), last(0.7840/0.1842)
m1_v4: adjust tail token arange, 2e-5 + 3 epoch, best(0.7963/2/0.3017), last(0.7863, 0.1991) -> ok
m1_v5: lr decay to 0.93, best(0.8007/2/0.2927), last(0.7890, 0.1809)
m1_v6: dropout to 0.3, best(0.7921/3/0.3232) -> ok
m1_v6: 2e-5 + 5 epcoh, best(0.7946/2/0.3159), last(0.7752, 0.0656) -> ok
m1_v7: lr to 1e-5 + 5 epoch, best(0.7972/2/0.3331) -> ok
m1_v8: weight2-1-1, 5 epoch, best(0.7805/2/0.3738), last(0.7719, 0.1503)
m1_v8: weight2-1-1, 2e-5 + 3 epoch, best(0.7984/2/0.3248), last(0.7783, 0.2055) -> ok
m1_v9: weight1.5-1-1, 2e-5 + 3 epoch, best(0.7877/2/0.3159), last(0.7803, 0.2018)
m1_v10: head 140 + tail 140, 2e-5 + 3epoch, best(0.7987/2/0.2937), last(0.7947/0.1867)

5 fold experiment:
dr -> 0.3 improve

token_id  7959  8179  8000  7891  8188  8043  8088
dr-0.3  7969  8188  8008  7919  8189  8055  8108
ML:510  7978  8097  8137  7929  8149  8058  -
dr+ML+1e-5  8064  8154  8075  7913  8166  8074  8064
dr+ML+2e-5  8048  8218  8176  7895  8178  8103  8108


best model:
robert_l24  submit_20191019_1023.csv  0.81028026000  0.804879 fixed     +   ^
robert_l12  submit_20191017_1601.csv  0.80695093000  0.806693 fixed
wwm_bert    submit_20191018_2228.csv  0.81096435000  0.810670 fixed     +   ^
xlnet_l12   submit_20191023_1026.csv  0.80419731000  0.808558 dynamic   +
robert_l24  submit_20191016_2232.csv  0.80771929000  0.807993 dynamic
xlnet_l12   submit_20191024_1020.csv  0.80685902000  0.809733 fixed         ^
