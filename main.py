import streamlit as st
import pandas as pd
import numpy as np
import datetime

chart_data = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])

# Add a timestamp column to the data
start_time = datetime.datetime.now()
timestamps = pd.date_range(start=start_time, periods=100, freq="5min")
chart_data["timestamp"] = timestamps


def main():
    st.title("명지 마트 재고 현황")
    st.write("301개 품목에 대한 재고 현황을 보여줍니다.")

    # Set the timestamp column as the index
    chart_data_with_index = chart_data.set_index("timestamp")

    multiselect = st.multiselect("Choose Item", ("A", "B", "C"))
    how_long = st.number_input(
        "How long?", min_value=10, max_value=100, value=20, step=1
    )

    line_chart = st.line_chart(chart_data_with_index[:how_long])

    st.dataframe(chart_data_with_index)


if __name__ == "__main__":
    main()
