from typing import Optional, List, Tuple, Any
import pandas as pd


class DataFilter:
    def __init__(self, filter_tuple_list: Optional[List[Tuple[str, Any]]]):

        self.filter_tuple_list = filter_tuple_list

    def get_name(self):

        if self.filter_tuple_list is None:
            return "all"
        filter_value_list = []
        for _filter_tuple_key, filter_tuple_value in self.filter_tuple_list:
            filter_value_list += [str(filter_tuple_value)]
        return str(",".join(filter_value_list))

    def filter_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:

        if self.filter_tuple_list is None:
            return df, 100

        df = df.copy()
        total_len = len(df)
        for filter_tuple_key, filter_tuple_value in self.filter_tuple_list:
            df = df[df[filter_tuple_key] == filter_tuple_value]
        return df, round(100 * len(df) / total_len)

    def get_filter_tuple_list(self):
        return self.filter_tuple_list
