from plotly.subplots import make_subplots
from math import floor, sqrt, ceil
import minisom
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import minisom
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import _tree
import re


def plot_som(som, data, target_col, max_obs=15):
    groups = list(target_col.unique())
    frac_arrays = np.tile(
        np.zeros(shape=som.get_euclidean_coordinates()
                 [1].flatten().shape).copy(),
        (len(groups), 1),
    )
    j = 0
    for x in data.values:
        som._activate(x)
        activated_neuron = som._activation_map.argmin()
        group_index = groups.index(target_col.iloc[j])
        frac_arrays[group_index][activated_neuron] = (
            frac_arrays[group_index][activated_neuron] + 1
        )
        j = j + 1
    ncols = ceil(sqrt(len(groups)))
    nrows = ceil(len(groups) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=(groups))

    j = 1
    k = 1

    for frac_array in frac_arrays:
        fig.add_trace(
            go.Scatter(
                x=som.get_euclidean_coordinates()[0].flatten(),
                y=som.get_euclidean_coordinates()[1].flatten(),
                mode="markers",
                hovertext=pd.Series(
                    som.activation_response(data.values).flatten()),
                text=pd.Series(
                    pd.Series(
                        np.nan_to_num(
                            (
                                frac_array
                                / som.activation_response(data.values).flatten()
                            )
                        )
                    )
                ),
                hovertemplate="BMU for %{hovertext:,.0f} of observations<br>"
                + "Share of observations in group: %{text:,.2f}<br>"
                + "X: %{x:,.0f}<br>"
                + "Y: %{y:,.0f}",
                marker=dict(
                    size=som.activation_response(data.values).flatten() / 2,
                    color=np.nan_to_num(
                        (frac_array / som.activation_response(data.values).flatten())
                    ),
                    colorscale="Viridis",
                    cmin=np.nan_to_num(
                        frac_arrays /
                        som.activation_response(data.values).flatten()
                    )[
                        :,
                        som.activation_response(
                            data.values).flatten() > max_obs,
                    ]
                    .flatten()
                    .min(),
                    cmax=np.nan_to_num(
                        frac_arrays /
                        som.activation_response(data.values).flatten()
                    )[
                        :,
                        som.activation_response(
                            data.values).flatten() > max_obs,
                    ]
                    .flatten()
                    .max(),
                ),
            ),
            row=j,
            col=k,
        )
        k = 1 + (k) % ncols
        if k == 1:
            j = j + 1
    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale="Viridis",
            cmin=np.nan_to_num(
                frac_arrays / som.activation_response(data.values).flatten()
            )[
                :,
                som.activation_response(data.values).flatten() > max_obs,
            ]
            .flatten()
            .min(),
            cmax=np.nan_to_num(
                frac_arrays / som.activation_response(data.values).flatten()
            )[
                :,
                som.activation_response(data.values).flatten() > max_obs,
            ]
            .flatten()
            .max(),
            colorbar=dict(
                title="Share of the observations from given group",  # title here
                titleside="right",
                tickfont=dict(family="Arial", size=14),
                titlefont=dict(size=20, family="Arial"),
            ),
        ),
    )

    fig.add_trace(colorbar_trace, row=1, col=1)
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showticklabels=True),
        yaxis=dict(showticklabels=True),
        autosize=False,
        width=1200,
        height=800,
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1,
                     # , showline=True, linecolor='grey'
                     gridcolor='grey', zeroline=True, zerolinewidth=2, zerolinecolor='grey'
                     )
    fig.update_yaxes(showgrid=True, gridwidth=1,
                     # , showline=True, linecolor='grey'
                     gridcolor='grey', zeroline=True, zerolinewidth=2, zerolinecolor='grey'
                     )
    fig.update_annotations(font_size=20)
    fig.show()


def som_get_important_variants(
    data,
    cols,
    som,
    column="group_mii",
    group="(-0.001, 3.0]",
    n_variants=5,
    neuron_coord=(3, 0),
):
    return (
        data.loc[
            np.intersect1d(
                som.win_map(data.loc[:, cols].values, True)[neuron_coord],
                data.loc[(data.loc[:, column] == group)].index.to_list(),
            ),
            cols,
        ]
        .sum(axis=0)
        .sort_values(ascending=False)
        .argsort()[:n_variants]
        .index.values
    )


def get_characteristic_neurons(
    data, cols, min_freq=0.6, min_obs=5, separate=True, target_col="publication_group"
):
    freq_table = pd.DataFrame(
        [
            data.groupby([target_col, "SOM_neuron"])["__id__"].count()
            / data.groupby([target_col, "SOM_neuron"])["__id__"]
            .count()
            .groupby(level=1)
            .sum(),
            data.groupby([target_col, "SOM_neuron"])["process_number"].count(),
        ]
    ).transpose()
    if separate:
        important_neurons = freq_table.loc[
            (freq_table["__id__"].values > min_freq)
            & (freq_table["process_number"].values > min_obs)
        ].index.droplevel([0])
        data["searched_node"] = data.loc[:, "SOM_neuron"].copy()
        data.iloc[
            np.where(
                np.isin(
                    data["searched_node"],
                    np.setdiff1d(data["SOM_neuron"].unique(),
                                 important_neurons.values),
                )
            )[0],
            data.columns.get_loc("searched_node"),
        ] = "not_important_neurons"
    else:
        important_neurons = freq_table.loc[
            (freq_table["__id__"].values > min_freq)
            & (freq_table["process_number"].values > min_obs)
        ].index
        mapper = dict(
            zip(
                important_neurons.droplevel([0]).values,
                important_neurons.droplevel([1]).values,
            )
        )
        data["searched_node"] = data.loc[:, "SOM_neuron"].copy()
        data.loc[:, "searched_node"] = data.loc[:, "searched_node"].map(mapper)
        data.loc[
            pd.isna(data["searched_node"]), "searched_node"
        ] = "not_important_neurons"
    data, paths = get_paths(data, cols, "searched_node", min_obs)

    return data, paths


def get_rules(tree, feature_names, class_names, min_sample_size=10):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    splits = []
    for path in paths:
        rule = "if "
        # print(np.argmax(path[-1][0][0]))
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        samples_size = path[-1][1]
        rule += f" | based on {samples_size:,} samples"
        if (samples_size > min_sample_size) & (
            class_names[l] != "not_important_neurons"
        ):
            rules += [rule]

    return rules


def get_paths(data, cols, target_col, min_obs):
    dt = DecisionTreeClassifier(max_depth=13)
    dt.fit(data[cols], data[target_col])
    data["predicted_neuron"] = dt.predict(data[cols])
    return data, get_rules(dt, cols, dt.classes_, min_obs)


def assign_neuron(data, som_data, som):
    indices = som.win_map(som_data.values, True)
    data.loc[:, "SOM_neuron"] = None
    for key in dict(indices).keys():
        data.iloc[indices[key], data.columns.get_loc("SOM_neuron")] = str(key)
    return data


def create_combinations_columns(data, path):
    new_cols = []
    for p in path:
        lvl0 = re.sub("and|if|then |\(|\)", "", p)
        lvl1 = " ".join(lvl0[: lvl0.find("proba")].split())
        lvl2 = re.sub("> 0.5", "1", re.sub("<= 0.5", "0", lvl1))
        columns = lvl2[: lvl2.find(" class: ")]
        selected_columns = columns.split(" ")[0::2]
        values = columns.split(" ")[1::2]
        values = [int(x) for x in values]
        new_col_name = columns.replace(" ", "_")
        data[new_col_name] = (
            (data.loc[:, selected_columns] == values).sum(axis=1)
            == len(selected_columns)
        ) * 1
        new_cols.append(new_col_name)
    return data, new_cols
