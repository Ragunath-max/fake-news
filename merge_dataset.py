import pandas as pd

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

fake_df = fake_df[["text", "label"]]
true_df = true_df[["text", "label"]]

news_df = pd.concat([fake_df, true_df], ignore_index=True)

news_df = news_df.sample(frac=1).reset_index(drop=True)

news_df.to_csv("news.csv", index=False)

print("news.csv created with REAL and FAKE labels")
print(news_df["label"].value_counts())
