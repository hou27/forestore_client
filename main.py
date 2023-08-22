import streamlit as st
import pandas as pd
import numpy as np

from last_timestamp import get_last_timestamp
from data import get_predictions


@st.cache_data()  # 캐싱 추가
def get_chart_data():
    chart_data = get_predictions()
    # chart_data = get_predictions(is_test=False)
    last_timestamp = get_last_timestamp(chart_data)

    last_timestamp_dataframe = pd.DataFrame(last_timestamp, columns=["timestamp"])
    last_timestamp_dataframe["item_id"] = [i for i in range(1, len(last_timestamp) + 1)]
    last_timestamp_dataframe = last_timestamp_dataframe.set_index("item_id")
    last_timestamp_dataframe.columns = ["last_timestamp"]

    return chart_data, last_timestamp_dataframe


def show_filtered_charts(lookback_range, chart_data, timestamps, selected_items=None):
    if selected_items is None:
        selected_items = chart_data.columns

    filtered_data = chart_data[list(map(lambda x: x, selected_items))]
    filtered_last_timestamp = timestamps.iloc[
        list(map(lambda x: str(int(x) - 1), selected_items))
    ]

    chart = st.empty()  # 준비된 요소 생성
    chart_dataframe = st.empty()  # 준비된 요소 생성

    chart.line_chart(filtered_data[:lookback_range])
    st.write("마지막 시간과 재고량")
    chart_dataframe.dataframe(filtered_last_timestamp)
    st.write("전체 데이터")
    st.dataframe(filtered_data)


def main():
    st.title("명지 마트 재고 현황")
    st.write("301개 품목에 대한 재고 현황을 보여줍니다.")

    chart_data, last_timestamp = get_chart_data()

    selected_items = st.multiselect("Choose Item", chart_data.columns)
    lookback_range = st.slider(
        "몇시간 뒤까지 보고 싶으신가요?", min_value=10, max_value=100, value=20, step=1
    )

    if selected_items:
        show_filtered_charts(
            lookback_range, chart_data, last_timestamp, selected_items
        )  # 변경되지 않은 부분
    else:
        show_filtered_charts(
            lookback_range, chart_data, last_timestamp, chart_data
        )  # 변경되지 않은 부분


if __name__ == "__main__":
    main()
