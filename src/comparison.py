import pandas as pd

fp16_df = pd.read_csv("./results/phi2_fp16_baseline.csv")
int8_df = pd.read_csv("./results/phi2_int8.csv")

comparison = pd.concat([fp16_df, int8_df])

print(comparison)