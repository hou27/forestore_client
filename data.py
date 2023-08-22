import pandas as pd
import datetime


# def get_mocking_data():
#     predictions_df = pd.read_excel("small_test_prediction_list.xlsx")
#     predictions = [
#         predictions_df.iloc[i].tolist() for i in range(1, len(predictions_df))
#     ]
#     item_ids = [i for i in range(1, 302)]
#     timestamp_base = datetime.datetime.now()

#     # Create timestamp list
#     timestamps = [
#         timestamp_base + datetime.timedelta(minutes=i * 60)
#         for i in range(len(predictions[0]))
#     ]

#     concatenated_data = {
#         f"{item_id}": values for item_id, values in zip(item_ids, predictions)
#     }

#     # Create a dataframe with timestamps as the index
#     df = pd.DataFrame(concatenated_data, index=timestamps)
#     return df


def get_predictions(is_test: bool = True):
    predictions_df = None

    if is_test:
        predictions_df = pd.read_excel("small_test_prediction_list.xlsx")
    else:
        predictions_df = pd.read_excel("prediction_list.xlsx")

    predictions = [
        predictions_df.iloc[i].tolist() for i in range(1, len(predictions_df))
    ]
    item_ids = [i for i in range(1, 302)]
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
