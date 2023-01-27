from typing import Optional, Dict
import pandas as pd
import numpy as np
import ipywidgets as widgets


class TabbedPlot:
    """IPy Tab Widget wrapper for plots

        In some cases plt.show() is neccessary  to function properly.

        Example:
            tabbed = TabbedPlot()
            for _ in range(5):
                with tabbed.add_plot():
                    plt.plot(np.random.uniform(size=100))
                    plt.show()
            tabbed.display()
    """
    def __init__(self) -> None:
        self._outputs = []
        self._titles = []
        self._tab_widget = widgets.Tab(children = self._outputs)
    
    def display(self):
        self._tab_widget = widgets.Tab(children = self._outputs)
        for idx, title in enumerate(self._titles):
            self._tab_widget.set_title(idx, title)
        display(self._tab_widget)

    def add_plot(self, tab_title: str = None):

        if tab_title is None: tab_title = f"Tab {len(self._outputs)}"
        self._titles.append(tab_title)

        self._outputs.append(widgets.Output())
        return self._outputs[-1]



def translate_label(
    label: str,
    label_translator: Optional[Dict[str, str]] = None,
):

    if label_translator is None:
        return label
    if label not in label_translator:
        return label
    return label_translator[label]


def dataframe_to_latex(df: pd.DataFrame):

    print("\\hline")
    print("", end="")
    for col in df.columns:
        print(f" & {col}", end="")
    print(" \\\\")

    for index, row in df.iterrows():

        print("\\hline")
        print(f"{index}", end="")
        for col in df.columns:
            value = row[col]
            if df[col].dtypes == np.dtype("int64"):
                value = int(row[col])
            if df[col].dtypes == np.dtype("float64"):
                value = "%.3f" % row[col]
            print(f" & {value}", end="")
        print(" \\\\")

    print("\\hline")
