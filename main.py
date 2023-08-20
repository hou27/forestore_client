import streamlit as st
import pandas as pd
import numpy as np
import datetime

st.title("명지 마트 재고 현황")

st.header("header")
st.subheader("subheader")

# draw graph
chart_data = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])

# Add a timestamp column to the data
start_time = datetime.datetime.now()
timestamps = pd.date_range(start=start_time, periods=100, freq="5min")
chart_data["timestamp"] = timestamps

# Create a remaining_stock column (use the 'a' column as an example)
chart_data["remaining_stock"] = chart_data["a"]

# Drop unwanted columns ('a', 'b', 'c')
chart_data.drop(["a", "b", "c"], axis=1, inplace=True)

st.dataframe(chart_data)

# st.markdown(
#     """
# **Markdown**
# # Heading 1
# ## Heading 2
# ### Heading 3
# :moon:<br>
# :star:<br>
# :sunglasses:<br>
# __Bold__
# _Italic_
# """,
#     unsafe_allow_html=True,
# )

multiselect = st.multiselect("Choose Item", ("A", "B", "C"))
how_long = st.number_input("How long?", min_value=10, max_value=100, value=20, step=1)

st.line_chart(chart_data.set_index("timestamp")[:how_long])
st.write(f"You selected: {multiselect}")

st.text("Plain text")
