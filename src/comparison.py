import pandas as pd

fp16_df = pd.read_csv("./results/phi2_fp16_baseline.csv")
int8_df = pd.read_csv("./results/phi2_int8.csv")
int4_df = pd.read_csv("./results/phi2_int4.csv")

comparison = pd.concat([fp16_df, int8_df, int4_df])

print(comparison)