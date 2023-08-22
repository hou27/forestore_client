import pandas as pd
import datetime

from product_info import product_info_dict


def get_predictions(is_test: bool = True):
    predictions_df = None

    if is_test:
        # predictions_df = pd.read_excel("small_test_prediction_list.xlsx")
        predictions_df = pd.read_excel("test_prediction_list.xlsx")
    else:
        predictions_df = pd.read_excel("prediction_list.xlsx")

    predictions = [
        predictions_df.iloc[i].tolist() for i in range(1, len(predictions_df))
    ]
    item_ids = [product_info_dict[i] for i in range(1, 302)]
    timestamp_base = datetime.datetime.now()

    # Create timestamp list
    timestamps = [
        timestamp_base + datetime.timedelta(minutes=i * 60)
        for i in range(len(predictions[0]))
    ]

    concatenated_data = {
        f"{item_id}": values for item_id, values in zip(item_ids, predictions)
    }

    # Create a dataframe with timestamps as the index
    df = pd.DataFrame(concatenated_data, index=timestamps)
    return df


def update_chart_data(chart_data, length_of_cycle):
    last_timestamp = []
    num_columns = len(chart_data.columns)

    for idx in range(num_columns):
        length = length_of_cycle[idx]
        column_data = chart_data.iloc[:, idx]

        if len(column_data) > 100:  # 범위 조건 (100)
            if column_data.iloc[length - 1] != 0:  # 길이 조건
                j = length

                while j < len(column_data):
                    if column_data.iloc[j - 1] - 5 < 0:
                        column_data.iloc[j] = 0
                        break
                    else:
                        column_data.iloc[j] = column_data.iloc[j - 1] - 5  # 이전 값에서 5 감소
                    j += 1
                column_data.iloc[j + 1 :] = 0  # j 이후의 값들을 모두 0으로 바꿈

        last_timestamp.append(chart_data.iloc[j].name)

    return chart_data, last_timestamp
