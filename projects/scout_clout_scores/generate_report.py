import seaborn as sns
import matplotlib.pyplot as plt
import pandas as  pd
from datetime import datetime
import plotly.graph_objects as go
import os
from os import walk
import glob
from pathlib import Path
from fpdf import FPDF

df_colors = pd.DataFrame({'colors': ['lightgray', 'lightgrey', 'pink',
                                     'pink', 'pink', 'pink',
                                     'lightgrey']})

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
y_limit = 100
x_limit = 2000


class PDF(FPDF):
    def __init__(self, timestamp):
        super().__init__()
        self.timestamp = timestamp
        self.WIDTH = 210
        self.HEIGHT = 297

    def header(self):
        # Custom logo and positioning
        # Create an `assets` folder and put any wide and short image inside
        # Name the image `logo.png`
        # self.image('assets/scout-blue.png', 10, 8, 33)
        self.set_font('Arial', 'B', 11)
        self.cell(self.WIDTH - 80)
        self.cell(60, 1, f'Report - {datetime.fromtimestamp(self.timestamp).strftime("%B %d, %Y")}', 0, 0, 'R')
        self.ln(20)

    def footer(self):
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def page_body(self, images):
        # Determine how many plots there are per page and set positions
        # and margins accordingly

        if len(images) == 3:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
            self.image(images[2], 15, self.WIDTH / 2 + 90, self.WIDTH - 30)
        elif len(images) == 2:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
        else:
            self.image(images[0], 15, 25, self.WIDTH - 30)

    def print_page(self, images):
        # Generates the report
        self.add_page()
        self.page_body(images)

def get_latest_timestamps(filenames):
    latest_timestamp = 0
    latest_fn = filenames[0]
    for fn in filenames:
        cur_time = int(fn.split('_')[-1].split('.')[0])
        if cur_time > latest_timestamp:
            latest_timestamp = cur_time
            latest_fn = fn
    return latest_timestamp, latest_fn

def get_latest_dfs(dir):
    _, _, all_score_csvs = next(walk(f'./{dir}'))
    latest_timestamp, latest_score_csv = get_latest_timestamps(all_score_csvs)
    _, _, all_failed_csvs = next(walk(f'./{dir}/failed'))
    _, latest_failed_csv = get_latest_timestamps(all_failed_csvs)
    return pd.read_csv(f'{dir}/{latest_score_csv}'), pd.read_csv(f'{dir}/failed/{latest_failed_csv}'), latest_timestamp


def generate_dsitribtuions(df_scout_scored, df_wtw_scored, timestamp):
    df_scores_scout = df_scout_scored['score']
    fig, axes = plt.subplots(nrows=2, figsize=(15, 12))

    fig.suptitle(datetime.fromtimestamp(timestamp).strftime("%B %d, %Y"), fontsize=20)

    sns.histplot(df_scores_scout, binwidth=50, color='#072241', ax=axes[0])
    for ax in axes.reshape(-1):
        ax.set_ylim(0, y_limit)
        ax.set_xlim(0, x_limit)
        ax.set_xlabel("Clout Score")
        ax.set_ylabel("Number of Users")

    fig.tight_layout(pad=2.5)

    axes[0].set_title(f'Scout Clout Score Distribution')
    axes[1].set_title(f'WTW Clout Score Distribution')

    df_scores_wtw = df_wtw_scored['score']
    sns.histplot(df_scores_wtw, binwidth=50, color='#5a2434', ax=axes[1])
    Path("./resources").mkdir(parents=True, exist_ok=True)
    plt.savefig('./resources/clout_dist.png')


def generate_density_functions():
    pass


def generate_stats_dict(scored_df, errors_df):
    error_count = errors_df.error.value_counts()
    try:
        no_posts = error_count["no posts"]
    except: 
        no_posts = 0

    stats = {"Number users registered": len(scored_df) + len(errors_df),
             "Number users scored": len(scored_df),
             "Number of profiles that could not be scored": len(errors_df),
             "Number of private profiles": error_count["private profile"],
             "Number of profiles that do not exist": error_count["profile does not exist"],
             "Number of profiles with no posts": no_posts,
             "Mean score": round(scored_df.describe().score["mean"], 2),
             "Median score": scored_df.describe().score["50%"],
             "Max score": scored_df.describe().score["max"]}
    return stats


def generate_stats_fig(scored_df, errors_df, figname='fig.png', title="Title"):
    stats = generate_stats_dict(scored_df, errors_df)
    fig = go.Figure(data=[go.Table(
        cells=dict(values=[list(stats.keys()), list(stats.values())], fill_color=[df_colors.colors], align='left'),
        columnwidth=[15, 5])
    ],
        layout=go.Layout(
            title=go.layout.Title(text=title),
            margin=dict(l=10, r=10, t=30, b=0)
        ))
    fig.update_layout(
        height=300,
        width=300,
        showlegend=False,
    )
    fig.write_image(f'./resources/{figname}', scale=3)


def generate_report():
    pass


def main():
    df_scout_score, df_scout_failed, timestamp = get_latest_dfs('scout_signups_history')
    df_wtw_score, df_wtw_failed, _ = get_latest_dfs('wtw_signups_history')
    generate_dsitribtuions(df_scout_score, df_wtw_score, timestamp)
    generate_stats_fig(df_scout_score, df_scout_failed, 'scout_stats.png', 'Scout Analysis')
    generate_stats_fig(df_wtw_score, df_wtw_failed, 'wtw_stats.png', 'WTW Analysis')

    pdf = PDF(timestamp)
    pdf.print_page(['./resources/clout_dist.png'])
    pdf.image("./resources/scout_stats.png", x=pdf.WIDTH / 2 - 80, y=180, w=80, h=80)
    pdf.image("./resources/wtw_stats.png", x=pdf.WIDTH / 2, y=180, w=80, h=80)
    Path("./reports").mkdir(parents=True, exist_ok=True)
    pdf.output(f'./reports/scout_report_{timestamp}.pdf', 'F')


if __name__ == "__main__":
    main()
