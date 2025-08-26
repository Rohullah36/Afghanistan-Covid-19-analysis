import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('free_outlier.csv')


# 1.line graph-COVID-19 Cases Over Time in Afghanistan

plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Date', y='Cases', color='red')
plt.title("COVID-19 Cases Over Time in Afghanistan", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Number of Cases")
plt.grid(alpha=0.3)
plt.show()


# 2.Ranked Bar Chart-Top 10 Provinces by COVID-19 Cases
top_provinces = df.groupby('Province')['Cases'].sum().nlargest(10)

plt.figure(figsize=(12,6))
sns.barplot(x=top_provinces.values, y=top_provinces.index, palette="Reds_r")
plt.title("Top 10 Provinces by COVID-19 Cases", fontsize=16)
plt.xlabel("Total Cases")
plt.ylabel("Province")
plt.show()


# 3. Pie chart: Share of Total Cases by Top 5 Provinces

top5_pie = df.groupby("Province")["Cases"].sum().sort_values(ascending=False).head(5)
plt.figure()
plt.pie(top5_pie.values, labels=top5_pie.index, autopct='%1.1f%%', startangle=140)
plt.title("Top 5 Provinces by Share of Cases")
plt.tight_layout()
plt.savefig("pie_chart_top5_provinces.png")


# 4. Stacked Bar Chart: Cases, Deaths, Recoveries by Top 5 Provinces

top5_provinces = df["Province"].value_counts().head(5).index
subset = df[df["Province"].isin(top5_provinces)]
summary = subset.groupby("Province")[["Cases", "Deaths", "Recoveries"]].sum()
summary.plot(kind="bar", stacked=True)
plt.title("Cases, Deaths, and Recoveries (Top Provinces)")
plt.ylabel("Count")
plt.xlabel('Province')
plt.tight_layout()
plt.savefig("stacked_bar_top5_provinces.png")