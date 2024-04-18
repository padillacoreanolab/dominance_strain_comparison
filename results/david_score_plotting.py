import pandas as pd
import math
import os
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
import xml.etree.ElementTree as ET

def split_by_strain(df):
    df = df.sort_values(by='strain')
    index = df['strain'].value_counts().iloc[0]
    df_strain1 = df.iloc[:index].copy()
    strain1 = df_strain1.at[1, 'strain']
    df_strain2 = df.iloc[index:].copy()
    strain2 = df_strain2.at[index+1, 'strain']
    df_array = [df_strain1, df_strain2]
    strain = [strain1, strain2]
    return df_array, strain

# function for the r-value correlation matrix for elo rating
def elo_pearson_correlation(df, input_directory):
    # sorting and splitting the file by strain
    split = split_by_strain(df)
    strain = split[1]
    df_array = split[0]
    a = 0
    # making files 
    for i in df_array:    
        i = i.drop(columns='strain')
        i = i.filter(like='ELO', axis=1)
        i.columns = i.columns.str.replace('_', ' ')
        i.columns = [col[:-4] for col in i.columns]
        df = i.corr()
        df_rounded = df.round(4)
        df_rounded.index.name = ''
        df_rounded.to_excel(input_directory + '//' + strain[a] + '_elo_r_value_matrix' + '.xlsx')
        
        # initializing plot size and font size
        plt.figure(figsize=(9.75, 7.5))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # Create the custom diverging palette
        if strain[a] == 'CD1':
            custom_palette = sns.diverging_palette(50, 230, as_cmap=True, center="light", s=100, l=30)
        if strain[a] == 'C57':
            custom_palette = sns.diverging_palette(235, 55, n=7, as_cmap=True, center="light", s=100, l=70)
        heatmap1 = sns.heatmap(df, annot=True, annot_kws={"size": 20}, cbar_kws={"label": "r-value"}, 
                            cmap=custom_palette, fmt=".4f", center=0, vmin=-0.7, vmax=0.7)
        cbar = plt.gca().collections[0].colorbar
        cbar.set_label("r-value", fontsize=22)
        cbar.ax.tick_params(labelsize=20)
        plt.xticks(rotation=0)
        for tick in heatmap1.get_xticklabels():
            tick.set_rotation(0)

        plt.title(strain[a] + " Dominance score correlation", fontsize=24, pad=24)
        
        # setting axis labels
        current_xtick_labels = [tick.get_text() for tick in heatmap1.get_xticklabels()]
        current_ytick_labels = [tick.get_text() for tick in heatmap1.get_yticklabels()]
        new1_xtick_labels = [label if label != "Home cage" else "Agonistic behavior" for label in current_xtick_labels]
        heatmap1.set_xticklabels(new1_xtick_labels)
        new_xtick_labels = [label if label != "Reward competition" else "Reward comp" for label in new1_xtick_labels]
        # Set the new tick labels for the x-axis
        heatmap1.set_xticklabels(new_xtick_labels)
        new1_ytick_labels = [label if label != "Home cage" else "Agonistic behavior" for label in current_ytick_labels]
        heatmap1.set_yticklabels(new1_ytick_labels)
        new_ytick_labels = [label if label != "Reward competition" else "Reward comp" for label in new1_ytick_labels]
        # Set the new tick labels for the y-axis
        heatmap1.set_yticklabels(new_ytick_labels)
        final_xtick_labels = [tick.get_text() for tick in heatmap1.get_xticklabels()]
        final_ytick_labels = [tick.get_text() for tick in heatmap1.get_yticklabels()]
        
        # splitting axis labels so they are in two lines
        x = 0
        for labels in new_xtick_labels:
            if len(labels.split()) > 1:
                result = '\n'.join(labels.split())
                final_xtick_labels[x] = result
            x += 1
        y = 0
        for labels in new_ytick_labels:
            if len(labels.split()) > 1:
                result = '\n'.join(labels.split())
                final_ytick_labels[y] = result
            y += 1
        heatmap1.set_xticklabels(final_xtick_labels)
        heatmap1.set_yticklabels(final_ytick_labels)
        plt.tight_layout()
        heatmap1.get_figure().savefig(input_directory + "//" + strain[a] + "_elo_r_value_heatmap.svg", format="svg")
        a += 1

# p-value matrix for elo rating
def elo_p_value(df, input_directory):
    # sorting and splitting the file by strain
    split = split_by_strain(df)
    strain = split[1]
    df_array = split[0]
    
    a = 0
    for i in df_array:
        i = i.drop(columns='strain')
        i = i.filter(like='ELO', axis=1)
        i.columns = i.columns.str.replace('_', ' ')
        i.columns = [col[:-4] for col in i.columns]
        # creating a dataframe for p_value matrix for strain 1
        columns1 = []
        column_names = list(i.columns)
        columns1.extend(column_names)
        p_value_df = pd.DataFrame(columns=columns1)
        for n in column_names:
            for m in column_names:    
                valid_indices = ~np.isnan(i[n]) & ~np.isnan(i[m])
                x = i[n][valid_indices]
                y = i[m][valid_indices]
                res = stats.pearsonr(x, y)
                p_value_df.loc[n, m] = res[1]
        df_numeric = p_value_df.apply(pd.to_numeric, errors='coerce')
        # creating a dataframe for p_value matrix for strain 2 
        # p_value_df = df_numeric.round(4)
        p_value_df.index.name = ''
        p_value_df.to_excel(input_directory + '//' + strain[a] + '_elo_p_value_matrix' + '.xlsx')
        plt.figure(figsize=(9.75, 7.5))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # custom_palette = sns.diverging_palette(240, 10, sep=20, n=7, as_cmap=True, center="light")
        if strain[a] == 'CD1':
            custom_palette = sns.diverging_palette(50, 230, as_cmap=True, center="light", s=100, l=30)
        if strain[a] == 'C57':
            # Create the custom diverging palette
            custom_palette = sns.diverging_palette(235, 55, n=7, as_cmap=True, center="light", s=100, l=70)
            # custom_palette = sns.diverging_palette(240, 60, as_cmap=True, center="light")
        heatmap2 = sns.heatmap(df_numeric, annot=True, annot_kws={"size": 20}, cbar_kws={"label": "p-value"},
                            cmap=custom_palette, fmt=".4f", center=0, vmin=-0.7, vmax=0.7)
        
        # sig_levels = [(0.05, '*'), (0.01, '**'), (0.001, '***')]
        # plt.text(j + 0.5, i + 0.5, symbol, horizontalalignment='center', verticalalignment='center')
        """b = 0
        for i in column_names:
            c = 0
            for j in column_names:
                if i != j:
                    p_value = df_numeric.loc[i, j]
                    if p_value <= 0.001:
                        plt.text(c + 0.85 , b + 0.35, '***', horizontalalignment='center', verticalalignment='center', fontsize=18)
                    elif p_value <= 0.01:
                        plt.text(c + 0.85, b + 0.35, '**', horizontalalignment='center', verticalalignment='center', fontsize=18)
                    elif p_value <= 0.05:
                        plt.text(c + 0.85, b + 0.35, '*', horizontalalignment='center', verticalalignment='center', fontsize=18)
                c += 1
            b += 1"""
        cbar = plt.gca().collections[0].colorbar
        cbar.set_label("p-value", fontsize=20)
        cbar.ax.tick_params(labelsize=20)
        # heatmap1.set_aspect("auto")
        plt.xticks(rotation=0)
        for tick in heatmap2.get_xticklabels():
            tick.set_rotation(0)
            # tick.set_ha('right')
        
        current_xtick_labels = [tick.get_text() for tick in heatmap2.get_xticklabels()]
        current_ytick_labels = [tick.get_text() for tick in heatmap2.get_yticklabels()]
        new_xtick_labels = [label if label != "Home cage" else "Agonistic behavior" for label in current_xtick_labels]
        # Set the new tick labels for the x-axis
        heatmap2.set_xticklabels(new_xtick_labels)
        new_ytick_labels = [label if label != "Home cage" else "Agonistic behavior" for label in current_ytick_labels]
        # Set the new tick labels for the y-axis
        heatmap2.set_yticklabels(new_ytick_labels)
        final_xtick_labels = [tick.get_text() for tick in heatmap2.get_xticklabels()]
        final_ytick_labels = [tick.get_text() for tick in heatmap2.get_yticklabels()]
        x = 0
        for labels in new_xtick_labels:
            if len(labels.split()) > 1:
                result = '\n'.join(labels.split())
                final_xtick_labels[x] = result
            x += 1
        y = 0
        for labels in new_ytick_labels:
            if len(labels.split()) > 1:
                result = '\n'.join(labels.split())
                final_ytick_labels[y] = result
            y += 1
        heatmap2.set_xticklabels(final_xtick_labels)
        heatmap2.set_yticklabels(final_ytick_labels)
        plt.title(strain[a] + " Dominance score correlation(p-value)", fontsize=24, pad=24)
        plt.tight_layout()
        heatmap2.get_figure().savefig(input_directory + "//" + strain[a] + "_elo_p_value_heatmap.svg")
        # plt.savefig(input_directory + "//" + strain[a] + "_p_value_heatmap.png")
        # plt.show()
        a += 1  

# pearson correlation coefficient between assays with strain.
def ds_pearson_correlation(df, input_directory):
    # sorting and splitting the file by strain
    split = split_by_strain(df)
    strain = split[1]
    df_array = split[0]
    
    a = 0
    for i in df_array:    
        i = i.drop(columns='strain')
        i = i.filter(like='DS', axis=1)
        i.columns = i.columns.str.replace('_', ' ')
        i.columns = [col[:-3] for col in i.columns]
        df = i.corr()
        df_rounded = df.round(4)
        df_rounded.index.name = ''
        df_rounded.to_excel(input_directory + '//' + strain[a] + '_ds_r_value_matrix' + '.xlsx')
        plt.figure(figsize=(9.75, 7.5))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        if strain[a] == 'CD1':
            custom_palette = sns.diverging_palette(50, 230, as_cmap=True, center="light", s=100, l=30)
        if strain[a] == 'C57':
            # Create the custom diverging palette
            custom_palette = sns.diverging_palette(235, 55, n=7, as_cmap=True, center="light", s=100, l=70)
            # custom_palette = sns.diverging_palette(240, 60, as_cmap=True, center="light")
    
        heatmap1 = sns.heatmap(df, annot=True, annot_kws={"size": 20}, cbar_kws={"label": "r-value"}, 
                            cmap=custom_palette, fmt=".4f", center=0, vmin=-0.7, vmax=0.7)
        # setting up the color bar
        cbar = plt.gca().collections[0].colorbar
        cbar.set_label("r-value", fontsize=22)
        cbar.ax.tick_params(labelsize=20)
        plt.xticks(rotation=0)
        for tick in heatmap1.get_xticklabels():
            tick.set_rotation(0)
        plt.title(strain[a] + " David Score Correlation", fontsize=24, pad=24)
        
        # making changes to axis labels
        current_xtick_labels = [tick.get_text() for tick in heatmap1.get_xticklabels()]
        current_ytick_labels = [tick.get_text() for tick in heatmap1.get_yticklabels()]
        new1_xtick_labels = [label if label != "Home cage" else "Agonistic behavior" for label in current_xtick_labels]
        heatmap1.set_xticklabels(new1_xtick_labels)
        new_xtick_labels = [label if label != "Reward competition" else "Reward comp" for label in new1_xtick_labels]
        # Set the new tick labels for the x-axis
        heatmap1.set_xticklabels(new_xtick_labels)
        new1_ytick_labels = [label if label != "Home cage" else "Agonistic behavior" for label in current_ytick_labels]
        heatmap1.set_yticklabels(new1_ytick_labels)
        new_ytick_labels = [label if label != "Reward competition" else "Reward comp" for label in new1_ytick_labels]
        # Set the new tick labels for the y-axis
        heatmap1.set_yticklabels(new_ytick_labels)
        final_xtick_labels = [tick.get_text() for tick in heatmap1.get_xticklabels()]
        final_ytick_labels = [tick.get_text() for tick in heatmap1.get_yticklabels()]
        x = 0
        for labels in new_xtick_labels:
            if len(labels.split()) > 1:
                result = '\n'.join(labels.split())
                final_xtick_labels[x] = result
            x += 1
        y = 0
        for labels in new_ytick_labels:
            if len(labels.split()) > 1:
                result = '\n'.join(labels.split())
                final_ytick_labels[y] = result
            y += 1
        heatmap1.set_xticklabels(final_xtick_labels)
        heatmap1.set_yticklabels(final_ytick_labels)
        plt.tight_layout()
        heatmap1.get_figure().savefig(input_directory + "//" + strain[a] + "_ds_r_value_heatmap.svg", format="svg")
        a += 1    

# code currently only accounts for 2 strains need to update
def p_value_calculation(df, input_directory):
    df = df.sort_values(by='strain')
    df_ordered = df.reset_index(drop=True)
    df_ordered.to_excel(input_directory + '//' + 'strain_ordered' + '.xlsx')
    index = df['strain'].value_counts().iloc[0]
    df_strain1 = df.iloc[:index].copy()
    strain1 = df_strain1.at[1, 'strain']
    df_strain2 = df.iloc[index:].copy()
    strain2 = df_strain2.at[index+1, 'strain']
    df_array = [df_strain1, df_strain2]
    strain = [strain1, strain2]
    
    # determine strain for name
    a = 0
    for i in df_array:
        i = i.drop(columns='strain')
        i = i.filter(like='DS', axis=1)
        i.columns = i.columns.str.replace('_', ' ')
        i.columns = [col[:-2] for col in i.columns]
        # creating a dataframe for p_value matrix for strain 1
        columns1 = []
        column_names = list(i.columns)
        columns1.extend(column_names)
        p_value_df = pd.DataFrame(columns=columns1)
        for n in column_names:
            for m in column_names:    
                valid_indices = ~np.isnan(i[n]) & ~np.isnan(i[m])
                x = i[n][valid_indices]
                y = i[m][valid_indices]
                res = stats.pearsonr(x, y)
                p_value_df.loc[n, m] = res[1]
        df_numeric = p_value_df.apply(pd.to_numeric, errors='coerce')
        # creating a dataframe for p_value matrix for strain 2 
        p_value_df = df_numeric.round(4)
        p_value_df.index.name = ''
        p_value_df.to_excel(input_directory + '//' + strain[a] + '_ds_p_value_matrix' + '.xlsx')
        plt.figure(figsize=(9.75, 7.5))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # custom_palette = sns.diverging_palette(240, 10, sep=20, n=7, as_cmap=True, center="light")
        if strain[a] == 'CD1':
            custom_palette = sns.diverging_palette(50, 230, as_cmap=True, center="light", s=100, l=30)
        if strain[a] == 'C57':
            custom_palette = sns.diverging_palette(235, 55, n=7, as_cmap=True, center="light", s=100, l=70)
            # custom_palette = sns.diverging_palette(240, 60, as_cmap=True, center="light")
        lower_color = custom_palette(0)  # Color at the extreme low end of the palette
        upper_color = custom_palette(256)  # Color at the extreme high end of the palette
        colors = [
            (lower_color, 0.0),
            (lower_color, 0.4),
            # Transition using the custom palette
            *[(custom_palette(i), 0.4 + i/640) for i in range(257)],
            (upper_color, 0.6), 
            (upper_color, 1.0),
        ]
        heatmap2 = sns.heatmap(df_numeric, annot=True, annot_kws={"size": 20}, cbar_kws={"label": "p-value"},
                            cmap=colors, fmt=".4f", center=0, vmin=-0.7, vmax=0.7)
        #heatmap1 = sns.heatmap(df, annot=True, annot_kws={"size": 20}, cbar_kws={"label": "r-value"}, 
        #                   cmap="vlag", fmt=".4f", center=0, vmin=-1, vmax=1)
        """b = 0
        for i in column_names:
            c = 0
            for j in column_names:
                if i != j:
                    p_value = df_numeric.loc[i, j]
                    if p_value <= 0.001:
                        plt.text(c + 0.85 , b + 0.35, '***', horizontalalignment='center', verticalalignment='center', fontsize=18)
                    elif p_value <= 0.01:
                        plt.text(c + 0.85, b + 0.35, '**', horizontalalignment='center', verticalalignment='center', fontsize=18)
                    elif p_value <= 0.05:
                        plt.text(c + 0.85, b + 0.35, '*', horizontalalignment='center', verticalalignment='center', fontsize=18)
                c += 1
            b += 1"""
        cbar = plt.gca().collections[0].colorbar
        cbar.set_label("p-value", fontsize=20)
        cbar.ax.tick_params(labelsize=20)
        # heatmap1.set_aspect("auto")
        plt.xticks(rotation=0)
        for tick in heatmap2.get_xticklabels():
            tick.set_rotation(0)
            # tick.set_ha('right')
       
        current_xtick_labels = [tick.get_text() for tick in heatmap2.get_xticklabels()]
        current_ytick_labels = [tick.get_text() for tick in heatmap2.get_yticklabels()]
        new_xtick_labels = [label if label != "Home Cage" else "Agonistic Behavior" for label in current_xtick_labels]
        # Set the new tick labels for the x-axis
        heatmap2.set_xticklabels(new_xtick_labels)
        new_ytick_labels = [label if label != "Home Cage" else "Agonistic Behavior" for label in current_ytick_labels]
        # Set the new tick labels for the y-axis
        heatmap2.set_yticklabels(new_ytick_labels)
        final_xtick_labels = [tick.get_text() for tick in heatmap2.get_xticklabels()]
        final_ytick_labels = [tick.get_text() for tick in heatmap2.get_yticklabels()]
        x = 0
        for labels in new_xtick_labels:
            if len(labels.split()) > 1:
                result = '\n'.join(labels.split())
                final_xtick_labels[x] = result
            x += 1
        y = 0
        for labels in new_ytick_labels:
            if len(labels.split()) > 1:
                result = '\n'.join(labels.split())
                final_ytick_labels[y] = result
            y += 1
        heatmap2.set_xticklabels(final_xtick_labels)
        heatmap2.set_yticklabels(final_ytick_labels)
        plt.title(strain[a] + " David Score Correlation(p-value)", fontsize=24, pad=24)
        plt.tight_layout()
        heatmap2.get_figure().savefig(input_directory + "//" + strain[a] + "_ds_p_value_heatmap.svg")
        a += 1  

"""def add_significance(input_directory):
    files = os.listdir(input_directory)

    def parse_svg_heatmap(svg_file):
        tree = ET.parse(svg_file)
        root = tree.getroot()
        
        # Example: Extracting data from SVG rectangles
        data = []
        for rect in root.findall(".//{http://www.w3.org/2000/svg}rect"):
            x = float(rect.attrib.get("x"))
            y = float(rect.attrib.get("y"))
            width = float(rect.attrib.get("width"))
            height = float(rect.attrib.get("height"))
            value = float(rect.attrib.get("data-value"))  # Example: Extracting a data attribute
            data.append({"x": x, "y": y, "width": width, "height": height, "value": value})
        
        # Create DataFrame from extracted data
        df = pd.DataFrame(data)
        return df
    # Pair up files
    file_pairs = list(combinations(files,2))
    for pair in file_pairs:
        # dataframe_pair = [None, None]
        for file in pair:
            if "p_value" in file: 
                dataframe_p = parse_svg_heatmap(file)
                # dataframe_pair[0] = dataframe_p
            else:
                dataframe_r = parse_svg_heatmap(file)
                plt.plot(dataframe_r)
                # dataframe_pair[1] = dataframe_r
            for row in dataframe_p.values:
                for value in row:
                    if value <= 0.001:
                        pass
                    elif p_value <= 0.01:
                        plt.text(c + 0.85, b + 0.35, '**', horizontalalignment='center', verticalalignment='center', fontsize=18)
                    elif p_value <= 0.05:
                        plt.text(c + 0.85, b + 0.35, '*', horizontalalignment='center', verticalalignment='center', fontsize=18)"""

def elo_r_value(df, input_directory):
        # sorting and splitting the file by strain
    split = split_by_strain(df)
    strain = split[1]
    df_array = split[0]
    
    a = 0
    for i in df_array:    
        i = i.drop(columns='strain')
        i = i.filter(like='DS', axis=1)
        i.columns = i.columns.str.replace('_', ' ')
        i.columns = [col[:-3] for col in i.columns]
        df = i.corr()
        df_rounded = df.round(4)
        df_rounded.index.name = ''
        df_rounded.to_excel(input_directory + '//' + strain[a] + '_ds_r_value_matrix' + '.xlsx')
        plt.figure(figsize=(9.75, 7.5))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        if strain[a] == 'CD1':
            custom_palette = sns.diverging_palette(50, 230, as_cmap=True, center="light", s=100, l=30)
        if strain[a] == 'C57':
            # Create the custom diverging palette
            custom_palette = sns.diverging_palette(235, 55, n=7, as_cmap=True, center="light", s=100, l=70)
            # custom_palette = sns.diverging_palette(240, 60, as_cmap=True, center="light")
    
        heatmap1 = sns.heatmap(df, annot=True, annot_kws={"size": 20}, cbar_kws={"label": "r-value"}, 
                            cmap=custom_palette, fmt=".4f", center=0, vmin=-0.7, vmax=0.7)
        # setting up the color bar
        cbar = plt.gca().collections[0].colorbar
        cbar.set_label("r-value", fontsize=22)
        cbar.ax.tick_params(labelsize=20)
        plt.xticks(rotation=0)
        for tick in heatmap1.get_xticklabels():
            tick.set_rotation(0)
        plt.title(strain[a] + " David Score Correlation", fontsize=24, pad=24)
        
        # making changes to axis labels
        current_xtick_labels = [tick.get_text() for tick in heatmap1.get_xticklabels()]
        current_ytick_labels = [tick.get_text() for tick in heatmap1.get_yticklabels()]
        new1_xtick_labels = [label if label != "Home cage" else "Agonistic behavior" for label in current_xtick_labels]
        heatmap1.set_xticklabels(new1_xtick_labels)
        new_xtick_labels = [label if label != "Reward competition" else "Reward comp" for label in new1_xtick_labels]
        # Set the new tick labels for the x-axis
        heatmap1.set_xticklabels(new_xtick_labels)
        new1_ytick_labels = [label if label != "Home cage" else "Agonistic behavior" for label in current_ytick_labels]
        heatmap1.set_yticklabels(new1_ytick_labels)
        new_ytick_labels = [label if label != "Reward competition" else "Reward comp" for label in new1_ytick_labels]
        # Set the new tick labels for the y-axis
        heatmap1.set_yticklabels(new_ytick_labels)
        final_xtick_labels = [tick.get_text() for tick in heatmap1.get_xticklabels()]
        final_ytick_labels = [tick.get_text() for tick in heatmap1.get_yticklabels()]
        x = 0
        for labels in new_xtick_labels:
            if len(labels.split()) > 1:
                result = '\n'.join(labels.split())
                final_xtick_labels[x] = result
            x += 1
        y = 0
        for labels in new_ytick_labels:
            if len(labels.split()) > 1:
                result = '\n'.join(labels.split())
                final_ytick_labels[y] = result
            y += 1
        heatmap1.set_xticklabels(final_xtick_labels)
        heatmap1.set_yticklabels(final_ytick_labels)
        plt.tight_layout()
        heatmap1.get_figure().savefig(input_directory + "//" + strain[a] + "_ds_r_value_heatmap.svg", format="svg")
        a += 1    

    
    

# obtain input directory with masterfile and plot all data
input_directory = input("directory from David Score calculation: ")
filename = "Master_file.csv"
file_path = os.path.join(input_directory, filename)
final_df = pd.read_csv(file_path, header=0)
ds_pearson_correlation(final_df, input_directory)
p_value_calculation(final_df, input_directory)
elo_pearson_correlation(final_df, input_directory)
elo_p_value(final_df, input_directory)

# combining files for a master correlation matrix file
# List to store DataFrames from individual files
dfs = []

# Iterate through each file in the directory
for file in os.listdir(input_directory):
    if file.endswith('.xlsx') and 'matrix' in file.lower():
        file_path = os.path.join(input_directory, file)
        # Read each Excel file into a DataFrame
        df = pd.read_excel(file_path)
        dfs.append((file.split('.')[0], df)) 
        os.remove(os.path.join(input_directory, file))
# Check if any files were found
if not dfs:
    print("No files found matching the criteria.")
else:
    output_file_path = input_directory + '//combined_correlation' + '.xlsx'
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        # Iterate through each DataFrame and write to a separate sheet
        for sheet_name, df in dfs:
            df.to_excel(writer, sheet_name=sheet_name, index=False)