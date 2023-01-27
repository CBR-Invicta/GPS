import prince
from scipy import sparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns


def row_coordinates(ca, X):
    """The row principal coordinates."""
    ca._check_is_fitted()

    if isinstance(X, pd.DataFrame):
        try:
            X = X.sparse.to_coo().astype(float)
        except AttributeError:
            X = X.to_numpy().astype(
                float
            )  # w tym wierszu jest błąd, brakowało zwykłego .astype(float)

    if ca.copy:
        X = X.copy()

    # Normalise the rows so that they sum up to 1
    if isinstance(X, np.ndarray):
        X = X / X.sum(axis=1)[:, None]
    else:
        X = X / X.sum(axis=1)

    return pd.DataFrame(
        data=X @ sparse.diags(ca.col_masses_.to_numpy() ** -0.5) @ ca.V_.T
    )


def column_coordinates(ca, X):
    """The column principal coordinates."""
    ca._check_is_fitted()

    if isinstance(X, pd.DataFrame):
        is_sparse = X.dtypes.apply(pd.api.types.is_sparse).all()
        if is_sparse:
            X = X.sparse.to_coo()
        else:
            X = X.to_numpy().astype(
                float
            )  # Tutaj jest błąd w kodzie źródłowym, dlatego przepisuję funkcję

    if ca.copy:
        X = X.copy()

    # Transpose and make sure the rows sum up to 1
    if isinstance(X, np.ndarray):
        X = X.T / X.T.sum(axis=1)[:, None]
    else:
        X = X.T / X.T.sum(axis=1)

    return pd.DataFrame(
        data=X @ sparse.diags(ca.row_masses_.to_numpy() ** -0.5) @ ca.U_
    )


def plot_CA(
    wspolrzedne_indeks,
    wspolrzedne_kolumny,
    ranking_selected_cols,
    analiza_korespondencji_agg,
):
    fig = go.Figure(
        data=go.Scatter(
            y=wspolrzedne_kolumny[1],
            x=wspolrzedne_kolumny[0],
            mode="markers+text",
            name="Variant",
            text=ranking_selected_cols,
            textfont=dict(size=18),
            marker=dict(size=20),
            textposition="top center"
        )
    )
    fig.add_trace(
        go.Scatter(
            y=wspolrzedne_indeks[1],
            x=wspolrzedne_indeks[0],
            mode="markers+text",
            name="Patient group",
            text=analiza_korespondencji_agg.index.to_list(),
            textfont=dict(size=18),
            marker=dict(size=20),
            textposition="top center"
        )
    )
    fig.update_layout(legend=dict(
        font=dict(
            # family="Courier",
            size=18,
            # color="black"
        )
    ),
        legend_title=dict(
        font=dict(
            # family="Courier",
            size=18,
            # color="blue")
        )
    )
    )
    fig.update_layout(
        xaxis=dict(title='Dimenstion 1 (63%)'),
        yaxis=dict(title='Dimenstion 2 (20%)'),
        autosize=False,
        width=1200,
        height=900,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        font=dict(
            family='Arial',
            size=18,
            color="black"
        ),
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
    # fig.update_xaxes(visible=False)
    # fig.update_yaxes(visible=False)
    fig.show()


def prepare_CA_data(input_data, sel_columns, sel_index="patient_group"):
    analiza_korespondencji_agg = pd.DataFrame(
        index=input_data[sel_index].unique(), columns=sel_columns
    )
    for indeks in analiza_korespondencji_agg.index:
        analiza_korespondencji_agg.loc[indeks] = input_data.loc[
            input_data[sel_index] == indeks, sel_columns
        ].sum(axis=0)
    return analiza_korespondencji_agg


def summarize_group(
    data,
    columns_to_summarize,
    group_col="patient_group",
    group="4-control_multiple_donors",
):
    if group != "":
        data_to_describe = data.loc[data[group_col]
                                    == group, columns_to_summarize]
    else:
        data_to_describe = data.loc[:, columns_to_summarize]
    g = sns.PairGrid(data_to_describe)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.histplot, kde=True)
    return data_to_describe.describe()


def ca_select_important_variants(wspolrzedne_indeks, wspolrzedne_kolumny, n_variants=5):
    return wspolrzedne_kolumny.index[
        np.linalg.norm(
            wspolrzedne_kolumny.values - wspolrzedne_indeks.loc[0, :].values, axis=1
        ).argsort()[:n_variants]
    ].values
