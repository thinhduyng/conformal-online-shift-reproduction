"""Script to reproduce Figure D.6: Stock Open Prices over time.

Shows historical open prices for Nvidia, AMD, BlackBerry, and Fannie Mae in a 2x2 grid.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from aci_repro.data_stocks import load_all_stocks
from aci_repro.plotting import use_style, savefig


def plot_figure_d6():
    use_style()
    stocks = load_all_stocks()
    
    # 2x2 grid matching paper layout: Nvidia, AMD, BlackBerry, Fannie Mae
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    
    order = [
        ("Nvidia", axes[0, 0]),
        ("AMD", axes[0, 1]),
        ("BlackBerry", axes[1, 0]),
        ("Fannie Mae", axes[1, 1]),
    ]
    
    for name, ax in order:
        df = stocks[name]
        # Only plot from index 1 onwards where returns are defined, matching author's indexing
        plot_df = df.iloc[1:]
        
        ax.plot(plot_df["Date"], plot_df["Open"], color="black", linewidth=1.5)
        ax.set_ylabel("Open Price ($)")
        
        # Inset text annotation for stock name
        ax.text(
            0.05, 0.90, name,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            verticalalignment="top"
        )
        
        if name == "Fannie Mae":
            ax.set_ylim(0, 100)
            
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=0.3)
        
    # X label on bottom subplots
    axes[1, 0].set_xlabel("Time")
    axes[1, 1].set_xlabel("Time")
    
    plt.tight_layout()
    
    # Save as both stockPrices and fig_d6_stock_prices
    savefig(fig, "stockPrices")
    savefig(fig, "fig_d6_stock_prices")
    plt.close(fig)
    print("Figure D.6 generated successfully in figures/")


if __name__ == "__main__":
    plot_figure_d6()
