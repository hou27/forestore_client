import streamlit as st
import pandas as pd
import numpy as np

from last_timestamp import get_last_timestamp
from data import get_predictions, update_chart_data
from product_info import product_info_dict


@st.cache_data()  # 캐싱 추가
def get_chart_data():
    chart_data = get_predictions()
    # chart_data = get_predictions(is_test=False)
    length_of_cycle = get_last_timestamp(chart_data)
    chart_data, last_timestamp = update_chart_data(chart_data, length_of_cycle)

    last_timestamp_dataframe = pd.DataFrame(last_timestamp, columns=["timestamp"])
    last_timestamp_dataframe["item_id"] = [
        product_info_dict[i] for i in range(1, len(last_timestamp) + 1)
    ]
    last_timestamp_dataframe = last_timestamp_dataframe.set_index("item_id")
    last_timestamp_dataframe.columns = ["last_timestamp"]

    return chart_data, last_timestamp_dataframe


def show_filtered_charts(lookback_range, chart_data, timestamps, selected_items=None):
    if selected_items is None:
        selected_items = chart_data.columns

    filtered_data = chart_data[list(map(lambda x: x, selected_items))]
    filtered_last_timestamp = timestamps.loc[list(map(lambda x: x, selected_items))]

    chart = st.empty()  # 준비된 요소 생성

    chart.line_chart(filtered_data[:lookback_range])
    st.markdown(
        """
    <p>예상 재고 소진 시점</p>
    """,
        unsafe_allow_html=True,
    )
    chart_dataframe = st.empty()  # 준비된 요소 생성
    chart_dataframe.dataframe(filtered_last_timestamp)
    st.markdown(
        """
    <p>전체 데이터를 보고 싶으시다면 아래 버튼을 눌러주세요.</p>
    """,
        unsafe_allow_html=True,
    )
    # 버튼 초기 상태 설정
    if "toggle" not in st.session_state:
        st.session_state["toggle"] = False

    # 버튼을 누르면 상태가 반전됨
    if st.button("전체 데이터 보기"):
        st.session_state["toggle"] = not st.session_state["toggle"]

    # 상태에 따라 요소를 표시 또는 숨김
    if st.session_state["toggle"]:
        st.dataframe(filtered_data)
    else:
        pass


def main():
    st.title("명지 마트 재고 현황")

    chart_data, last_timestamp = get_chart_data()

    st.write(f"{len(chart_data.columns)}개 품목에 대한 재고 현황을 보여줍니다.")

    # 초기 값은 last_timestamp값이 가장 빠른 시점인 3개
    init_selected_items = last_timestamp.sort_values(by="last_timestamp").index[:3]

    selected_items = st.multiselect(
        "품목을 선택하세요 (초기 항목들은 예상 재고 소진 시점이 가장 이른 항목들입니다.)", chart_data.columns
    )
    lookback_range = st.slider(
        "몇시간 뒤까지 보고 싶으신가요?", min_value=10, max_value=100, value=20, step=1
    )

    if selected_items:
        show_filtered_charts(
            lookback_range, chart_data, last_timestamp, selected_items
        )  # 변경되지 않은 부분
    else:
        show_filtered_charts(
            lookback_range, chart_data, last_timestamp, init_selected_items
        )  # 변경되지 않은 부분


if __name__ == "__main__":
    main()
