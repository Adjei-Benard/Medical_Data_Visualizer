# medical_data_visualizer.py

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

# 1. Import data (simulated here for demonstration)
df = pd.DataFrame({
    'id': range(1, 101),
    'age': np.random.randint(29*365, 65*365, 100),
    'height': np.random.randint(150, 200, 100),
    'weight': np.random.randint(50, 120, 100),
    'ap_hi': np.random.randint(100, 180, 100),
    'ap_lo': np.random.randint(60, 120, 100),
    'cholesterol': np.random.randint(1, 4, 100),
    'gluc': np.random.randint(1, 4, 100),
    'smoke': np.random.randint(0, 2, 100),
    'alco': np.random.randint(0, 2, 100),
    'active': np.random.randint(0, 2, 100),
    'cardio': np.random.randint(0, 2, 100)
})

# 2. Add overweight column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3. Normalize cholesterol and gluc
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4-7. Categorical plot
def draw_cat_plot():
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    fig = sns.catplot(
        data=df_cat, 
        x='variable', 
        y='total', 
        hue='value', 
        col='cardio', 
        kind='bar'
    ).fig

    fig.savefig(os.path.join(output_dir, "catplot.png"))
    return fig

# 10-16. Heatmap
def draw_heat_map():
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    corr = df_heat.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", center=0, square=True, cbar_kws={"shrink": 0.5})
    fig.savefig(os.path.join(output_dir, "heatmap.png"))
    return fig

# Execute both plots
if __name__ == '__main__':
    draw_cat_plot()
    draw_heat_map()
    df.to_csv(os.path.join(output_dir, "processed_medical_data.csv"), index=False)  # save processed data
