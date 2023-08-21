import streamlit as st
import pandas as pd
import numpy as np

from last_timestamp import get_last_timestamp
from mocking_data import get_mocking_data

chart_data = get_mocking_data()
last_timestamp = get_last_timestamp(chart_data)

# 마지막 timestamp와 재고량 출력을 위한 dataframe 생성
last_timestamp = pd.DataFrame(last_timestamp, columns=["timestamp"])
last_timestamp["item_id"] = [i for i in range(1, len(last_timestamp) + 1)]
last_timestamp = last_timestamp.set_index("item_id")
last_timestamp.columns = ["last_timestamp"]


def main():
    st.title("명지 마트 재고 현황")
    st.write("301개 품목에 대한 재고 현황을 보여줍니다.")

    multiselect = st.multiselect("Choose Item", (i for i in range(1, 302)))
    how_long = st.number_input(
        "How long?", min_value=10, max_value=100, value=20, step=1
    )

    filtered_data = chart_data[list(map(lambda x: str(x), multiselect))]
    filtered_last_timestamp = last_timestamp.iloc[
        list(map(lambda x: str(x - 1), multiselect))
    ]

    if multiselect:
        line_chart = st.line_chart(filtered_data[:how_long])

        # 마지막 timestamp와 재고량 출력
        st.write("마지막 시간과 재고량")
        st.dataframe(filtered_last_timestamp)

        st.dataframe(filtered_data)
    else:
        line_chart = st.line_chart(chart_data[:how_long])

        # 마지막 timestamp와 재고량 출력
        st.write("마지막 시간과 재고량")
        st.dataframe(last_timestamp)

        st.dataframe(chart_data)


if __name__ == "__main__":
    main()
