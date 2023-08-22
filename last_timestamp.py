# 재고량이 0이 되는 시점을 계산하는 함수
import math


def get_last_timestamp(df):
    length_of_cycle = []
    for i in range(len(df.columns)):
        for j in range(1, len(df)):
            if (
                (df.iloc[j, i] > df.iloc[j - 1, i] + 5 and df.iloc[j - 1, i] <= 10)
                or df.iloc[j, i] <= 0
                or j == len(df) - 1
                or df.iloc[j, i] > df.iloc[j - 1, i] + 20
            ):
                length_of_cycle.append(j)
                break

    return length_of_cycle
