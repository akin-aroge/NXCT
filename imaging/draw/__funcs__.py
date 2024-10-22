from matplotlib.patches import Circle
import matplotlib.pyplot as plt

def draw_circle(center, radius, ax=None, edgecolor=None, facecolor='None', linewidth=3):

    if ax is None:
        ax = plt.gca()

    ax.add_patch(Circle(xy=center, radius=radius, edgecolor=edgecolor,
                        facecolor=facecolor, linewidth=linewidth))

