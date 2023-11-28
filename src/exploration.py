from tabulate import tabulate
import pandas as pd

matches = pd.read_csv("../data/raw/atp_matches_till_2022.csv")

german_ioc_code = "GER"
german_winner_matches = matches[matches["winner_ioc"] == german_ioc_code]
german_loser_matches = matches[matches["loser_ioc"] == german_ioc_code]

siege = german_winner_matches["loser_ioc"].value_counts()
niederlagen = german_loser_matches["winner_ioc"].value_counts()

performance = pd.DataFrame(
    {"Siege": siege, "Niederlagen": niederlagen}).fillna(0)

performance["Total"] = performance.sum(axis=1)
performance["Win_Rate"] = performance["Siege"] / performance["Total"]
performance["Loss_Rate"] = performance["Niederlagen"] / performance["Total"]
performance = performance[performance["Total"] >=
                          10].sort_values(by="Win_Rate", ascending=False)

top = 10
top_favoriten = performance.head(top)
top_rivalen = performance.tail(top)

print(f'Top {top} der besten {top_favoriten.columns[3]}')
print(tabulate(top_favoriten, headers='keys', tablefmt='presto'))
print(f'Top {top} der schlechtesten {top_rivalen.columns[3]}')
print(tabulate(top_rivalen.sort_values(by="Win_Rate",
      ascending=True), headers='keys', tablefmt='presto'))
