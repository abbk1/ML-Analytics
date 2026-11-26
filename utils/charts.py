"""Consistent analytical charts; input sampling is explicit in the page UI."""
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

COLORS = ['#2463eb', '#0e938d', '#d97706', '#9333ea', '#dc2626', '#475569']


def color_for(label):
    return COLORS[int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) % len(COLORS)]


def show(fig, key=None):
    fig.update_layout(template='plotly_white', font={'family': 'Arial, sans-serif', 'size': 13},
                      margin={'l': 25, 'r': 25, 't': 45, 'b': 25}, legend_title_text='')
    st.plotly_chart(fig, width='stretch', key=key, config={
        'displaylogo': False, 'toImageButtonOptions': {'format': 'png', 'scale': 2}})


def class_chart(distribution, title='Class distribution'):
    fig = px.bar(distribution, x='class', y='count', text='count', title=title,
                 color='class', color_discrete_sequence=COLORS)
    fig.update_yaxes(rangemode='tozero')
    show(fig)


def confusion_chart(result, normalized=False):
    matrix = result['evaluation']['confusion_matrix']
    if matrix is None:
        st.info('This import contains no confusion matrix.')
        return
    if normalized:
        totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)
    fig = px.imshow(matrix, x=result['classes'], y=result['classes'], text_auto='.2f' if normalized else True,
                    color_continuous_scale='Blues', labels={'x': 'Predicted label', 'y': 'Actual label', 'color': 'Proportion' if normalized else 'Count'})
    show(fig)


def response_curves(result, kind):
    fig = go.Figure()
    for label, data in result['evaluation'].get('curves', {}).items():
        x, y = (data['fpr'], data['tpr']) if kind == 'ROC' else (data['recall'], data['precision'])
        fig.add_trace(go.Scatter(x=x, y=y, name=str(label), mode='lines', line={'color': color_for(str(label))}))
    if not fig.data:
        st.info(f'{kind} curves are not applicable: no compatible continuous responses were supplied.')
        return
    fig.update_layout(xaxis_title='False positive rate' if kind == 'ROC' else 'Recall',
                      yaxis_title='True positive rate' if kind == 'ROC' else 'Precision', title=f'{kind} · one-vs-rest')
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    show(fig)
