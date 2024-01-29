import pandas as pd
import matplotlib.pyplot as plt

# Social Media Popularity
metrics = {'Candidate': ['Robert F. Kennedy Jr.', 'Cornel West', 'Jill Stein', 'Claudia de la Cruz'],
        'Twitter': [2683623, 1091212, 284395, 38740],
        'Instagram': [1600000,383000,53800,63500],
        'TikTok Followers': [599600,14400,2404,109100],
        'TikTok Hashtags': [2499,574,555,833]}

df = pd.DataFrame(metrics)


# Set up subplots for bar charts
fig_bar, axs_bar = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
axs_bar = axs_bar.flatten()

# Define custom colors for the bars
colors = ['skyblue', 'pink', 'gold', 'green']

# Iterate through each candidate for bar charts
for i in range(len(df)):
    ax = axs_bar[i]
    ax.bar(df.columns[1:], df.iloc[i, 1:], color=colors)
    ax.set_title(f"{df['Candidate'][i]}'s Social Media Metrics")
    ax.set_ylabel("Followers")

# Adjust layout
plt.tight_layout()
plt.show()