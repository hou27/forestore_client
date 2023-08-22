import plotly.graph_objects as go


def draw_graph(df):
    # 그래프 생성
    fig = go.Figure()

    # 라인 플롯 추가
    for column in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode="lines", name=column))

    # 현재 시점 마커 추가
    for i, column in enumerate(df.columns):
        name = column + " 현재 시점"

        fig.add_trace(
            go.Scatter(
                x=[df.index[0]],
                y=[df[column].iloc[0]],
                mode="markers",
                marker=dict(size=15, color=f"rgba({i*80},{i*40}, 255, .5)"),
                name=name,
                legendgroup=column,
            )
        )

    # 그래프 제목 및 축 레이블 설정
    fig.update_layout(
        title="시간에 따른 재고량",
        xaxis_title="시간",
        yaxis_title="재고량",
        legend=dict(
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=1.1,
            orientation="h",
        ),
    )

    return fig
