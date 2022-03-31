import sys
import string
import locale
import csv
import math
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import argparse
from typing import Tuple, Union
import pandas as pd
from pandas import read_csv
import numpy as np
from scipy.spatial import distance
import re
import random
import secrets


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, required=True)
args = parser.parse_args()


class IPv4:
    ipv4_pattern = '^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'

    def __init__(self) -> None:
        self.ip_cluster_ipv4 = {}
        self.ip_new_replace_or_ipv4 = {}
        self.ip_last_sum_ipv4 = {}
        self.new_ipcount_map_ipv4 = {}
        self.sorted_new_ipv4 = []
        self.sorted_new_ipmap_ipv4 = {}

    def validate_ipv4(self, ip_addr: str) -> bool:
        reg = re.compile(self.ipv4_pattern)
        return reg.search(str(ip_addr)) is not None

    def random_num(self, high: int, low: int) -> int:
        return random.randint(0, high - low) + low

    def call_ipv4_ops(self, df: pd.DataFrame, num_instances: int) -> pd.DataFrame:
        self.update_maps_ipv4(df)
        replaced_df = self.replaceor_ipv4(df)
        self.get_info_ipv4(replaced_df)
        df = self.aggregation_ipv4(replaced_df, num_instances)

        return df

    def update_maps_ipv4(self, df: pd.DataFrame) -> None:
        src_ip = df.iloc[:, 0]
        dest_ip = df.iloc[:, 1]
        for i in range(len(df) - 1):
            if (self.validate_ipv4(src_ip[i])):
                ip = src_ip[i].split('.')
                last = int(ip[len(ip) - 1])
                key = ''
                for j in range(len(ip) - 1):
                    key = key + ip[j] + "."

                if self.ip_cluster_ipv4.get(key) is not None:
                    self.ip_cluster_ipv4.update({key: self.ip_cluster_ipv4.get(key) + 1})
                else:
                    self.ip_cluster_ipv4.update({key: 1})
                pre = ''

                if self.ip_new_replace_or_ipv4.get(key) is not None:
                    pre = self.ip_new_replace_or_ipv4.get(key)
                else:
                    pre = str(random.randint(0, 254) + 1) + "." + \
                        str(random.randint(0, 254) + 1) + "." + \
                        str(random.randint(0, 254) + 1) + "."
                    self.ip_new_replace_or_ipv4.update({key: pre})

                if self.ip_last_sum_ipv4.get(key) is not None:
                    self.ip_last_sum_ipv4.update({key: int(self.ip_last_sum_ipv4.get(key)) + last})
                else:
                    self.ip_last_sum_ipv4.update({key: last})

            if (self.validate_ipv4(dest_ip[i])):
                ip = dest_ip[i].split('.')
                last = int(ip[len(ip) - 1])
                key = ''
                for j in range(len(ip) - 1):
                    key = key + ip[j] + "."

                if self.ip_cluster_ipv4.get(key) is not None:
                    self.ip_cluster_ipv4.update({key: self.ip_cluster_ipv4.get(key) + 1})
                else:
                    self.ip_cluster_ipv4.update({key: 1})
                pre = ''

                if self.ip_new_replace_or_ipv4.get(key) is not None:
                    pre = self.ip_new_replace_or_ipv4.get(key)
                else:
                    pre = str(self.random_num(210, 10)) + "." + \
                        str(self.random_num(210, 10)) + "." + str(self.random_num(210, 10)) + "."
                    self.ip_new_replace_or_ipv4.update({key: pre})

                if self.ip_last_sum_ipv4.get(key) is not None:
                    self.ip_last_sum_ipv4.update({key: int(self.ip_last_sum_ipv4.get(key)) + last})
                else:
                    self.ip_last_sum_ipv4.update({key: last})
        return

    def replaceor_ipv4(self, df: pd.DataFrame) -> pd.DataFrame:
        src_ip = df.iloc[:, 0]
        dest_ip = df.iloc[:, 1]
        for i in range(len(df)):
            new_src = ''
            new_dest = ''
            if (self.validate_ipv4(src_ip[i])):
                ip = src_ip[i].split('.')
                key = ''
                for j in range(len(ip) - 1):
                    key = key + ip[j] + "."

                pre = self.ip_new_replace_or_ipv4.get(key)
                num = self.ip_cluster_ipv4.get(key)
                sums = self.ip_last_sum_ipv4.get(key)
                mean = sums // num
                new_src = str(pre) + str(mean)
            else:
                new_src = src_ip[i]

            if (self.validate_ipv4(dest_ip[i])):
                ip = dest_ip[i].split('.')
                key = ''
                for j in range(len(ip) - 1):
                    key = key + ip[j] + "."

                pre = self.ip_new_replace_or_ipv4.get(key)
                num = self.ip_cluster_ipv4.get(key)
                sums = self.ip_last_sum_ipv4.get(key)
                mean = sums // num
                new_dest = str(pre) + str(mean)
            else:
                new_dest = dest_ip[i]

            df.iloc[i, 0] = new_src
            df.iloc[i, 1] = new_dest

        return df

    def get_info_ipv4(self, df_info) -> None:
        src_ip = df_info.iloc[:, 0]
        dest_ip = df_info.iloc[:, 1]
        for i in range(len(df_info) - 1):
            if (self.validate_ipv4(src_ip[i])):
                if src_ip[i] in self.new_ipcount_map_ipv4:
                    self.new_ipcount_map_ipv4.update(
                        {
                            src_ip[i]: self.new_ipcount_map_ipv4.get(src_ip[i]) + 1
                        }
                    )
                else:
                    self.new_ipcount_map_ipv4.update(
                        {
                            src_ip[i]: 1
                        }
                    )
                    self.sorted_new_ipv4.append(src_ip[i])

            if (self.validate_ipv4(dest_ip[i])):
                if dest_ip[i] in self.new_ipcount_map_ipv4:
                    self.new_ipcount_map_ipv4.update(
                        {
                            dest_ip[i]: self.new_ipcount_map_ipv4.get(dest_ip[i]) + 1
                        }
                    )
                else:
                    self.new_ipcount_map_ipv4.update({dest_ip[i]: 1})
                    self.sorted_new_ipv4.append(dest_ip[i])

            self.sorted_new_ipv4.sort()
            for j in range(len(self.sorted_new_ipv4)):
                self.sorted_new_ipmap_ipv4.update({self.sorted_new_ipv4[j]: j})

        return

    def aggregation_ipv4(self, df_replaced_arg: pd.DataFrame, num_instances: int) -> pd.DataFrame:
        src_ip = df_replaced_arg.iloc[:, 0].to_numpy()
        dest_ip = df_replaced_arg.iloc[:, 1].to_numpy()
        for i in range(len(df_replaced_arg) - 1):
            if (self.validate_ipv4(src_ip[i])):
                src_ip[i] = self.get_near_ipv4(src_ip[i], num_instances)

            if (self.validate_ipv4(dest_ip[i])):
                dest_ip[i] = self.get_near_ipv4(dest_ip[i], num_instances)
        df_replaced_arg.iloc[:, 0] = src_ip
        df_replaced_arg.iloc[:, 1] = dest_ip

        return df_replaced_arg

    def get_near_ipv4(self, ip: str, k: int) -> str:
        result = ip
        index = self.sorted_new_ipmap_ipv4.get(ip)
        count = self.new_ipcount_map_ipv4.get(ip)

        if (count < k):
            dis = 0
            for i in range(len(self.sorted_new_ipv4)):
                if (self.new_ipcount_map_ipv4.get(self.sorted_new_ipv4[i]) >= k):
                    dis = i - index
                    break
            for i in range(index - 1, 0):
                if (self.new_ipcount_map_ipv4.get(self.sorted_new_ipv4[i]) >= k):
                    if ((index - i) < dis):
                        dis = i - index
                        break
            newIndex = index + dis
            result = self.sorted_new_ipv4[newIndex]
        return result


class IPv6:
    ipv6_pattern = (
        '(?:^|(?<=\s))(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|'
        '([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4})'
        '{1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}'
        '(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]'
        '{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4})'
        '{0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}'
        '[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:('
        '(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}'
        '[0-9]))(?=\s|$)'
    )
    multicast_pattern = '^(FF00|ff00|Ff00|fF00)'
    unicast_ipv6 = 1
    multicast_ipv6 = 2

    def __init__(self) -> None:
        self.cluster_id_ipv6 = 1

        self.anon_list_ipv6 = []
        self.ip_prefix_count_ipv6 = {}
        self.ip_prefix_replace_or_ipv6 = {}
        self.ip_first_sum_ipv6 = {}
        self.ip_second_sum_ipv6 = {}
        self.ip_third_sum_ipv6 = {}
        self.ip_fourth_sum_ipv6 = {}
        self.ip_fifth_sum_ipv6 = {}
        self.cluster_idmap_ipv6 = {}
        self.prefix_maps_anon_ipv6 = {}
        self.ipv6_types = {}
        self.unclustered_ips_ipv6 = []

    def init_variables(self) -> None:
        self.anon_list_ipv6 = []
        self.ip_prefix_count_ipv6 = {}
        self.ip_prefix_replace_or_ipv6 = {}
        self.ip_first_sum_ipv6 = {}
        self.ip_second_sum_ipv6 = {}
        self.ip_third_sum_ipv6 = {}
        self.ip_fourth_sum_ipv6 = {}
        self.ip_fifth_sum_ipv6 = {}
        self.cluster_idmap_ipv6 = {}
        self.prefix_maps_anon_ipv6 = {}
        self.ipv6_types = {}
        self.unclustered_ips_ipv6 = []

    def validate_ipv6(self, ip_addr: str) -> bool:
        reg = re.compile(self.ipv6_pattern)
        return (reg.search(str(ip_addr)) is not None)

    def call_ipv6_ops(self, df: pd.DataFrame, num_instances: int) -> pd.DataFrame:

        self.init_variables()

        source_ips = df.iloc[:, 0]
        dest_ips = []

        for i in range(0, len(source_ips)):
            test_ip = source_ips[i]
            if self.validate_ipv6(test_ip):
                self.separate_ipv6(test_ip)

        self.create_anon_ipv6(num_instances)

        for i in range(0, len(self.anon_list_ipv6)):
            prefix = self.ip_prefix_replace_or_ipv6.get(self.anon_list_ipv6[i])
            nip = self.prefix_maps_anon_ipv6.get(prefix)
            df.iloc[i, 0] = nip

        if (len(df.columns) > 1):
            self.init_variables()
            dest_ips = df.iloc[:, 1]

            for i in range(0, len(dest_ips)):
                test_ip = dest_ips[i]
                if self.validate_ipv6(test_ip):
                    self.separate_ipv6(test_ip)

            self.create_anon_ipv6(num_instances)

            for i in range(0, len(self.anon_list_ipv6)):
                prefix = self.ip_prefix_replace_or_ipv6.get(self.anon_list_ipv6[i])
                nip = self.prefix_maps_anon_ipv6.get(prefix)
                df.iloc[i, 1] = nip

        return df

    def separate_ipv6(self, test_ip: str) -> None:
        ipv6_type = None
        anon_parts = 0
        non_anonymizable_ip = None
        anonymizable_ip = None

        ipv6_type, anon_parts = self.ipv6_type_detection(test_ip)

        test_ip = self.create_ip_string_ipv6(test_ip)
        non_anonymizable_ip, anonymizable_ip = self.separate_ip_parts_ipv6(test_ip, anon_parts)
        self.anon_list_ipv6.append(anonymizable_ip)
        self.ipv6_types.update({anonymizable_ip: ipv6_type})

        self.update_ipv6_maps(non_anonymizable_ip, anonymizable_ip, anon_parts)

    def ipv6_type_detection(self, ip_addr: str) -> Tuple[int, int]:
        if (re.search(self.multicast_pattern, ip_addr) is None):
            return self.unicast_ipv6, 3
        else:
            return self.multicast_ipv6, 4

    def create_ip_string_ipv6(self, ip_addr: str) -> str:
        parts = ip_addr.split(':')
        num_parts = len(parts)
        new_ipv6 = ''
        zero_parts = 8 - num_parts + 1

        for i in range(0, num_parts):
            if (len(parts[i]) == 0):
                for j in range(0, zero_parts):
                    new_ipv6 += '0:'
            else:
                new_ipv6 += (parts[i] + ':')

        new_ipv6 = new_ipv6[0: len(new_ipv6) - 1]
        return new_ipv6

    def separate_ip_parts_ipv6(self, ip_addr: str, anon_bit: int) -> Tuple[str, str]:
        non_anon_str = ''
        anon_str = ''
        parts = ip_addr.split(':')

        # separates the portion to be anonymized
        for i in range(0, anon_bit):
            anon_str += (parts[i] + ':')
        anon_str = anon_str[0: len(anon_str) - 1]

        # separates the portion to be averaged
        for i in range(anon_bit, 8):
            non_anon_str += (parts[i] + ':')
        non_anon_str = non_anon_str[0: len(non_anon_str) - 1]

        return non_anon_str, anon_str

    def update_ipv6_maps(self, preserved_ipaddr: str, anon_ipaddr: str, anon_parts: str) -> str:
        if self.ip_prefix_count_ipv6.get(anon_ipaddr) is not None:
            self.ip_prefix_count_ipv6.update(
                {
                    anon_ipaddr: self.ip_prefix_count_ipv6.get(anon_ipaddr) + 1
                }
            )
        else:
            self.ip_prefix_count_ipv6.update({anon_ipaddr: 1})

        pre = ''

        if self.ip_prefix_replace_or_ipv6.get(anon_ipaddr) is not None:
            pre = self.ip_prefix_replace_or_ipv6.get(anon_ipaddr)
        else:
            parts = anon_ipaddr.split(':')
            pre = parts[0]
            for i in range(1, anon_parts):
                pre += (':' + str(self.random_hexnum()))

            self.ip_prefix_replace_or_ipv6.update({anon_ipaddr: pre})

        non_anon = preserved_ipaddr.split(':')

        # update the maps that store the sum of the suffix parts
        self.create_suffix_replace_or_ipv6(self.ip_first_sum_ipv6, anon_ipaddr, non_anon[0])
        self.create_suffix_replace_or_ipv6(self.ip_second_sum_ipv6, anon_ipaddr, non_anon[1])
        self.create_suffix_replace_or_ipv6(self.ip_third_sum_ipv6, anon_ipaddr, non_anon[2])
        self.create_suffix_replace_or_ipv6(self.ip_fourth_sum_ipv6, anon_ipaddr, non_anon[3])

        if anon_parts == 3:  # ip type is unicast
            self.create_suffix_replace_or_ipv6(self.ip_fifth_sum_ipv6, anon_ipaddr, non_anon[4])

    def random_hexnum(self):
        return secrets.token_hex(2)

    def create_suffix_replace_or_ipv6(
            self, map: dict, key: Union[str, int], new_val: Union[str, int]) -> None:
        if map.get(key) is not None:
            update_val = int(map.get(key), 16) + int(new_val, 16)
            map.update({key: hex(update_val)[2:]})
        else:
            map.update({key: new_val})

    def create_anon_ipv6(self, num_instances: int) -> None:
        temp_dict = sorted(self.ip_prefix_count_ipv6.items(), key=lambda item: item[1])
        for i in temp_dict:
            self.unclustered_ips_ipv6.append(i[0])

        unclustered_ips_copy = self.unclustered_ips_ipv6.copy()
        ip_len = len(unclustered_ips_copy)
        for i in range(0, ip_len):
            ip = unclustered_ips_copy[i]
            if ip in self.unclustered_ips_ipv6:
                self.replaceor_ipv6(ip, num_instances)

    def replaceor_ipv6(self, anon_ipaddr: str, k: int) -> str:
        prefix = self.ip_prefix_replace_or_ipv6.get(anon_ipaddr)
        num = self.ip_prefix_count_ipv6.get(anon_ipaddr)
        ipv6_type = self.ipv6_types.get(anon_ipaddr)
        suffix = ''
        new_ip = ''

        self.unclustered_ips_ipv6.remove(anon_ipaddr)

        if num >= k:
            suffix = self.compute_mean_of_kipv6(anon_ipaddr, ipv6_type, num)
            new_ip = prefix + suffix
            self.prefix_maps_anon_ipv6.update({prefix: new_ip})
            self.cluster_idmap_ipv6.update({prefix: self.cluster_id_ipv6})

        else:
            list_to_make_k = []
            listlen = len(self.unclustered_ips_ipv6)

            for i in range(0, listlen):
                ip = self.unclustered_ips_ipv6[i]
                if ipv6_type == self.ipv6_types.get(ip):
                    list_to_make_k.append(ip)
                    num += self.ip_prefix_count_ipv6.get(ip)

                # the following conditional makes sure that only one group with less than k
                # instances is not left out
                if (i == listlen - 2):
                    ip1 = list(self.unclustered_ips_ipv6)[listlen - 1]
                    c = self.ip_prefix_count_ipv6.get(ip1)
                    if (c < k and ipv6_type == self.ipv6_types.get(ip1)):
                        num += c
                        list_to_make_k.append(ip1)
                        break

                if (num >= k):
                    break

            # remove the IPs that have been anonymized from the list,
            # so that they are not calculated again
            for i in range(0, len(list_to_make_k)):
                self.unclustered_ips_ipv6.remove(list_to_make_k[i])

            list_to_make_k.append(anon_ipaddr)

            suffix += (':' + self.compute_mean_multiple_ipv6(
                list_to_make_k, self.ip_first_sum_ipv6))
            suffix += (':' + self.compute_mean_multiple_ipv6(
                list_to_make_k, self.ip_second_sum_ipv6))
            suffix += (':' + self.compute_mean_multiple_ipv6(
                list_to_make_k, self.ip_third_sum_ipv6))
            suffix += (':' + self.compute_mean_multiple_ipv6(
                list_to_make_k, self.ip_fourth_sum_ipv6))
            if ipv6_type == self.unicast_ipv6:
                suffix += (':' + self.compute_mean_multiple_ipv6(
                    list_to_make_k, self.ip_fifth_sum_ipv6))

            new_ip = prefix + suffix

            for i in range(0, len(list_to_make_k)):
                pre = self.ip_prefix_replace_or_ipv6.get(list_to_make_k[i])
                self.prefix_maps_anon_ipv6.update({pre: (pre + suffix)})
                self.cluster_idmap_ipv6.update({pre: self.cluster_id_ipv6})

        self.cluster_id_ipv6 += 1

        return new_ip

    def compute_mean_of_kipv6(self, anon_ipaddr: str, ip_type: str, num: int) -> str:
        suffix = ''

        sums = self.ip_first_sum_ipv6.get(anon_ipaddr)
        mean = hex(int(sums, 16) // num)[2:]
        suffix += (':' + str(mean))

        sums = self.ip_second_sum_ipv6.get(anon_ipaddr)
        mean = hex(int(sums, 16) // num)[2:]
        suffix += (':' + str(mean))

        sums = self.ip_third_sum_ipv6.get(anon_ipaddr)
        mean = hex(int(sums, 16) // num)[2:]
        suffix += (':' + str(mean))

        sums = self.ip_fourth_sum_ipv6.get(anon_ipaddr)
        mean = hex(int(sums, 16) // num)[2:]
        suffix += (':' + str(mean))

        if ip_type == self.unicast_ipv6:
            sums = self.ip_fifth_sum_ipv6.get(anon_ipaddr)
            mean = hex(int(sums, 16) // num)[2:]
            suffix += (':' + str(mean))

        return suffix

    def compute_mean_multiple_ipv6(self, ips: list, map: dict) -> str:
        ips_len = len(ips)
        first = ips[0]
        num = self.ip_prefix_count_ipv6.get(first)
        sums = int(map.get(first), 16)
        for i in range(1, ips_len):
            ip = ips[i]
            num += self.ip_prefix_count_ipv6.get(ip)
            sums += int(map.get(ip), 16)

        mean = hex(sums // num)[2:]

        return str(mean)


class DataProcess:
    cut_threshold = 0.001

    def __init__(self) -> None:
        pass

    def is_validate_csv_file(self, filename: str) -> bool:
        try:
            with open(filename, newline='') as csvfile:
                start = csvfile.read(4096)

                # isprintable does not allow newlines, printable does not allow umlauts...
                if not all([c in string.printable or c.isprintable() for c in start]):
                    return False
                csv.Sniffer().sniff(start)
                return True
        except csv.Error:
            # Could not get a csv dialect -> probably not a csv.
            return False

    def read_dataset(self, filename: str, seperator: str, header_exists: bool) -> pd.DataFrame:
        df1 = pd.DataFrame()
        if header_exists:
            df1 = pd.read_csv(filename, sep=seperator)
        else:
            df1 = pd.read_csv(filename, sep=seperator, header=None)
        if (len(df1.columns) == 1):
            df1 = pd.read_csv(filename, sep="\t", header=None)
        return df1

    # generates dataframe and the list of fields that need to be anonymized based on contexts
    def create_df_to_work_on(
        self,
        df: pd.DataFrame,
        cols: list,
        col_id: str,
        use_context: bool
    ) -> Tuple[pd.DataFrame, list]:
        anon_fields = []
        cs = len(df.columns)
        if col_id:
            cols = cols + " " + col_id

        num_list = list(int(num) for num in cols.strip().split(' '))

        if use_context:
            total_count = len(df)
            # print(total_count)

            for i in range(0, cs):
                if i in num_list:
                    if (i == cs - 1 and col_id):
                        break

                    column = df.iloc[:, i]
                    unique_items = Counter(column)

                    for key in unique_items:
                        value = unique_items[key]
                        percentage = value / total_count

                        if (percentage < self.cut_threshold):
                            anon_fields.append(i)
                            break

            if col_id:
                anon_fields.append(int(col_id))

            df2 = pd.DataFrame()
            for i in anon_fields:
                df2 = pd.concat([df2, pd.DataFrame(df.iloc[:, i])], axis=1, ignore_index=True)

            return df2, anon_fields

        else:
            df1 = pd.DataFrame()
            for i in range(0, cs):
                if i in num_list:
                    # print('i = ',i)
                    df1 = pd.concat([df1, pd.DataFrame(df.iloc[:, i])], axis=1, ignore_index=True)
            return df1, num_list

    def condense_perclass(
        self,
        df11: pd.DataFrame,
        df_org: pd.DataFrame,
        epsilon: float,
        t: int,
        priv: bool,
        acc: bool,
        col_id: str,
        anon_fields: list
    ) -> None:
        # Anonymized source and destination IPs will be used
        # for displaying results to the user with the '.'
        src_ip_anonymized = df11.iloc[:, 0]
        dest_ip_anonymized = df11.iloc[:, 1]

        df1 = pd.DataFrame()

        # add a faked label column if no label in original data
        if not col_id:
            df11[len(df11.columns)] = 1

        # drop ip address, should be done after selecting columns because
        # otherwise column index in original data will be incorrect
        df1 = df11.drop(df11.columns[[0, 1]], axis=1)
        cs = len(df1.columns)
        rs = len(df1)
        print("new column length:" + str(cs))

        # class column should not be normalized
        normalized_data = pd.DataFrame(np.zeros((rs, cs - 1)))
        normalizer = []
        o_min_val = []
        o_max_val = []
        for i in range(cs - 1):
            o_min_val.append(np.min(df1.values[:, i]))
            o_max_val.append(np.max(df1.values[:, i]))
        rowidx = np.zeros(rs, dtype=np.int16)
        for i in range(rs):
            rowidx[i] = i

        df2 = pd.DataFrame(rowidx)
        df = pd.concat([df2, df1], axis=1, ignore_index=True)

        num_of_class = df.iloc[:, cs].max()

        print("num of class: " + str(num_of_class))
        csize = np.zeros(num_of_class, dtype=np.int16)
        for i in range(num_of_class):
            tmp1 = np.where(df.iloc[:, cs] == i + 1)
            csize[i] = len(tmp1[0])

        print("csize: " + str(csize))
        tres_csize = np.zeros(num_of_class, dtype=np.int)
        for i in range(0, num_of_class):
            tres_csize[i] = np.floor(csize[i] / int(t))

        print("tres_csize: " + str(tres_csize))
        size_of_subgrp = np.gcd(tres_csize[0], tres_csize[1])

        if num_of_class > 2:
            for i in range(1, num_of_class):
                size_of_subgrp = np.gcd(size_of_subgrp, tres_csize[i])

        size_of_subgrp = size_of_subgrp * t
        print("Cluster size: " + str(size_of_subgrp))
        # group_size = np.floor(rs / size_of_subgrp)

        result = pd.DataFrame()

        for i in range(num_of_class):
            temp_data = pd.DataFrame()
            cellVal = np.where(df.iloc[:, cs] == i + 1)
            temp_data = temp_data.append(df.iloc[cellVal[0], :], ignore_index=True)
            # temp_rs = len(temp_data)
            # group_size = np.floor(temp_rs / size_of_subgrp)
            tempres = self.modified_condensation_p2(temp_data, size_of_subgrp, epsilon)
            result = result.append(tempres, ignore_index=True)
            print('tempres len ' + str(len(tempres)))

        sorted_result = result.sort_values('row_id')
        row_id = sorted_result['row_id']
        sorted_result.drop(sorted_result.columns[0], axis=1, inplace=True)

        # equation for normalization: (x-P_min)/(P_max-P_min)*(O_max-O_min) + O_min
        # normalized Data
        for i in range(0, cs - 1):
            p_min_val = float((np.min(sorted_result.values[:, i])))
            p_max_val = float((np.max(sorted_result.values[:, i])))
            normalizer.append(p_max_val - p_min_val)
            if normalizer[i] > 0:
                normalized_data.values[:, i] = (
                    (sorted_result.values[:, i] - p_min_val) / (normalizer[i])) * (
                        o_max_val[i] - o_min_val[i]) + o_min_val[i]
            if normalizer[i] == 0:
                normalized_data.values[:, i] = 0

        normalized_data.to_csv('perturbed-normalized.txt', sep=',', index=False)

        sorted_result = pd.concat([dest_ip_anonymized, sorted_result], axis=1, ignore_index=True)
        sorted_result = pd.concat([src_ip_anonymized, sorted_result], axis=1, ignore_index=True)
        sorted_result = pd.concat([row_id, sorted_result], axis=1, ignore_index=True)

        sorted_result = sorted_result.sort_values(sorted_result.columns[0])
        df1 = pd.concat([dest_ip_anonymized, df1], axis=1, ignore_index=True)
        df1 = pd.concat([src_ip_anonymized, df1], axis=1, ignore_index=True)
        sorted_result.to_csv('perturbed-withrowid.txt', sep=',', index=False)

        sorted_resultnorowid = sorted_result.drop(sorted_result.columns[0], axis=1)
        sorted_resultnorowid.to_csv('perturbed.txt', sep=',', index=False)
        sorted_resultdataonly = sorted_result.iloc[:, 3:cs + 2]
        sorted_resultdataonly.to_csv('perturbed-dataonly.txt', sep=',', index=False, header=False)
        rawdata = df1.iloc[:, 2:cs + 1]

        rawdata.to_csv('rawdata.txt', sep=',', index=False, header=False)
        if priv:
            self.privacy('rawdata.txt', 'perturbed-dataonly.txt', 'privacy.txt', 0.95)

        """write file remake code here"""
        cs1 = len(df_org.columns)

        df2 = pd.DataFrame()
        df3 = pd.DataFrame()
        df_per = read_csv("perturbed.txt", sep=",")

        j = 0
        for i in range(0, cs1):
            if i in anon_fields:
                df2 = pd.concat([df2, pd.DataFrame(df_per.iloc[:, j])], axis=1, ignore_index=True)
                df3 = pd.concat([df3, pd.DataFrame(df11.iloc[:, j])], axis=1, ignore_index=True)
                j += 1
            else:
                df2 = pd.concat([df2, pd.DataFrame(df_org.iloc[:, i])], axis=1, ignore_index=True)
                df3 = pd.concat([df3, pd.DataFrame(df_org.iloc[:, i])], axis=1, ignore_index=True)

        df2.to_csv('remade_perturbed.txt', sep=',', index=False)

        if acc:
            if ip_version == 4:
                self.compareacc(df3, 'unsw-test-small.csv', 'remade_perturbed.txt')
            else:
                self.compareacc(df3, 'test-ipv6.csv', 'remade_perturbed.txt')

    def privacy(self, data_file: str, modified_file: str, output_file: str, conf: float) -> None:
        locale.setlocale(locale.LC_ALL, 'en_US.utf8')
        data = pd.read_csv(data_file, sep=",", header=None).to_numpy()
        nrows, ncols = data.shape

        modified_data = pd.read_csv(modified_file, sep=",", header=None).to_numpy()

        diffdata = modified_data - data
        diffdata_norm = np.copy(diffdata)

        ranges = []
        nrsme = np.zeros(ncols)
        nrsme2 = np.zeros(ncols)
        for i in range(0, ncols):
            ranges.append(float(np.max(data[:, i])) - float(np.min(data[:, i])))

            if ranges[i] > 0:
                stdi2 = np.std(data[:, i])
                stdi = np.ptp(data[:, i])
                diffdata[:, i] = diffdata[:, i] / stdi
                diffdata_norm[:, i] = diffdata_norm[:, i] / stdi2
                nrsme[i] = sqrt(mean_squared_error(data[:, i], modified_data[:, i])) / stdi
                nrsme2[i] = sqrt(mean_squared_error(data[:, i], modified_data[:, i])) / stdi2

        nrsme_mean = np.mean(nrsme)
        nrsme2_mean = np.mean(nrsme2)
        print("root mean squared error normalized min-max:" + str(nrsme_mean))
        print("root mean squared error normalized std:" + str(nrsme2_mean))
        upper = (1 + conf) / 2.0
        lower = (1 - conf) / 2.0
        privacy = np.zeros((4, ncols))

        privacy[0, :] = np.quantile(diffdata, upper, axis=0) - np.quantile(diffdata, lower, axis=0)
        privacy[3, :] = np.quantile(diffdata_norm, upper, axis=0) - \
            np.quantile(diffdata_norm, lower, axis=0)

        var_diff_data = np.var(diffdata, axis=0)
        var_data = np.var(data, axis=0)
        for y in range(len(var_diff_data)):
            if var_data[y] == 0:
                var_data[y] = 1
            privacy[1, y] = var_diff_data[y] / var_data[y]

        diffdata2 = np.abs(diffdata)

        privacy[2, :] = np.quantile(diffdata2, 0.5, axis=0)

        privacy_mean = privacy.mean(axis=1)
        print(privacy_mean)
        meanPrivacy = pd.DataFrame(privacy_mean)
        meanPrivacy.to_csv(output_file, sep=',')

        privacy_val = privacy_mean[0]
        privacy_val2 = privacy_mean[3]
        print("Privacy:" + str(privacy_val))
        print("Privacy normalized by std:" + str(privacy_val2))

    # accuracy checker code:
    def compareacc(self, dfTrain, test_file, perturbed_file):
        cs = len(dfTrain.columns)
        print(cs)
        dfTrainData = dfTrain.iloc[:, :cs - 1]
        dfTrainData = dfTrainData.drop(dfTrainData.columns[[0, 1]], axis=1)

        dfTrainLabel = dfTrain.iloc[:, cs - 1]
        np.savetxt('train-label.txt', dfTrainLabel, delimiter='\n')
        dfTrain2 = pd.read_csv(perturbed_file, sep=",")
        dfTrainData2 = dfTrain2.iloc[:, 0:cs - 1]
        dfTrainData2 = dfTrainData2.drop(dfTrainData2.columns[[0, 1]], axis=1)

        dfTrainLabel2 = dfTrain2.iloc[:, cs - 1]
        np.savetxt('train-label-2.txt', dfTrainLabel2, delimiter='\n')

        df_test = pd.read_csv(test_file, sep=",")
        if (len(df_test.columns) == 1):
            df_test = pd.read_csv(test_file, sep=",", header=None)
        df_test_data = df_test.iloc[:, :cs - 1]
        df_test_data = df_test_data.drop(df_test_data.columns[[0, 1]], axis=1)

        df_testLabel = df_test.iloc[:, cs - 1]
        np.savetxt('test-label.txt', df_testLabel, delimiter='\n')
        md1 = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
        md1.fit(dfTrainData, dfTrainLabel)

        c1 = md1.predict(df_test_data)
        np.savetxt('predicted-label.txt', c1, delimiter='\n')
        md2 = KNeighborsClassifier(n_neighbors=1, algorithm='brute')

        md2.fit(dfTrainData2, dfTrainLabel2)

        c2 = md2.predict(df_test_data)
        np.savetxt('perturbed-label.txt', c2, delimiter='\n')

        v1 = c1 == df_testLabel
        v2 = c2 == df_testLabel
        len_v1 = 0
        len_v2 = 0
        for i in range(len(v1)):
            if v1[i] is True:
                len_v1 = len_v1 + 1
        for i in range(len(v2)):
            if v2[i] is True:
                len_v2 = len_v2 + 1
        accuracyorig = len_v1 / len(df_test_data)
        accuracycondense = len_v2 / len(df_test_data)
        print("accuracyorig = " + str(accuracyorig))
        print("accuracycondense = " + str(accuracycondense))

        FP1 = 0
        TP1 = 0
        FN1 = 0
        TN1 = 0
        FP2 = 0
        TP2 = 0
        FN2 = 0
        TN2 = 0
        # 2 is attack, 1 is normal
        for i in range(len(df_testLabel)):
            if (df_testLabel[i] == 2 and c2[i] == 2):
                TP2 = TP2 + 1
            if (df_testLabel[i] == 2 and c2[i] == 1):
                FN2 = FN2 + 1
            if (df_testLabel[i] == 1 and c2[i] == 1):
                TN2 = TN2 + 1
            if (df_testLabel[i] == 1 and c2[i] == 2):
                FP2 = FP2 + 1
            if (df_testLabel[i] == 2 and c1[i] == 2):
                TP1 = TP1 + 1
            if (df_testLabel[i] == 2 and c1[i] == 1):
                FN1 = FN1 + 1
            if (df_testLabel[i] == 1 and c1[i] == 1):
                TN1 = TN1 + 1
            if (df_testLabel[i] == 1 and c1[i] == 2):
                FP1 = FP1 + 1

        tp_rate_orig = TP1 / (FN1 + TP1)
        fp_rate_orig = FP1 / (TN1 + FP1)
        tp_rate_cond = TP2 / (FN2 + TP2)
        fp_rate_cond = FP2 / (TN2 + FP2)

        print("Original True positive rate = " + str(tp_rate_orig))
        print("Original False positive rate = " + str(fp_rate_orig))
        print("Condense True positive rate = " + str(tp_rate_cond))
        print("COndense False positive rate = " + str(fp_rate_cond))

    def modified_condensation_p2(
        self, data: pd.DataFrame,
        group_size: int,
        epsilon: float
    ) -> pd.DataFrame:
        row_id = data.iloc[:, 0]
        data.drop(data.columns[0], axis=1, inplace=True)
        label = data.iloc[:, -1]
        data.drop(data.columns[-1], axis=1, inplace=True)
        mini = self.min(data)
        mini_array = mini.to_numpy()  # minimum dataframe to array
        data_array = data.to_numpy()  # input dataframe to array
        dis = distance.cdist(
            data_array, mini_array, 'euclidean')  # calculating distance from minimum
        dis = np.append(dis, data_array, axis=1)  # append distance to the data
        dis_df = pd.DataFrame(dis)  # data df with distance
        dis_df.insert(
            loc=0, column='row_id', value=row_id)  # insert column back(front of dataframe)
        dis_df.insert(
            loc=len(dis_df.columns),
            column='label',
            value=label
        )  # insert column back(end of dataframe)
        dis = dis_df.to_numpy()  # convert back to array for sorting
        sorted_array = dis[dis[:, 1].argsort()]  # sort array based on second column(distance)
        sorted_df = pd.DataFrame(sorted_array)
        distance_df = self.clustering(sorted_df, group_size)  # split data in clusters
        distance_df.rename(
            columns={distance_df.columns[0]: "row_id"},
            inplace=True
        )  # rename column 0 with row_id
        distance_df.rename(
            columns={distance_df.columns[1]: "distance"},
            inplace=True
        )  # rename column 1 with distance
        distance_df.rename(
            columns={distance_df.columns[len(distance_df.columns) - 2]: "label"},
            inplace=True
        )  # rename last column with label
        distance_df['row_id'] = distance_df['row_id'].astype(int)  # make row id column integers
        distance_df['Cluster'] = distance_df['Cluster'].astype(int)  # make cluster column integers
        mean_df = self.mean_calc(
            distance_df,
            group_size,
            epsilon)  # calculate mean, ptp for each cluster
        mean_df.drop(
            ['distance', 'Cluster'], axis=1, inplace=True)  # drops columns distance, cluster
        return mean_df

    # function to create clustering
    def clustering(self, sorted_df: pd.DataFrame, group_size: int) -> pd.DataFrame:
        rows1 = sorted_df.index
        rows = len(rows1)
        m = rows / group_size  # calculate number of clusters
        m = math.floor(m)
        m = int(m)

        if (m % 2) == 0:
            # calculate size of mid cluster
            middle_size = (
                rows - (m / 2 - 1) * group_size) - ((m / 2) * group_size + 1)
            df_array = sorted_df.to_numpy()  # dataframe to array
            clusters1 = np.repeat(np.arange(1, int(m / 2 + 1)), group_size)  # first part
            clusters1 = clusters1[:, None]  # reshape array
            clusters2 = np.repeat(int((m / 2 + 1)), middle_size + 1)  # mid part
            clusters2 = clusters2[:, None]
            clusters3 = np.repeat(np.arange(int((m / 2) + 2), m + 1), group_size)  # last part
            clusters3 = clusters3[:, None]
            cluster_array = np.concatenate(
                (clusters1, clusters2, clusters3), axis=None)  # put together the 3 arrays
            cluster_array = cluster_array[:, None]
            final_array = np.append(
                df_array, cluster_array, axis=1)  # final array with assigned clusters
            df_final = pd.DataFrame(final_array)  # final dataframe with assigned clusters
            df_final.columns = [*df_final.columns[:-1], 'Cluster']  # rename only last column
        else:
            middle_size = (rows - (m - 1) / 2 * group_size) - (
                (m - 1) / 2 * group_size + 1)  # calculate size of mid cluster
            df_array = sorted_df.to_numpy()  # dataframe to array
            clusters1 = np.repeat(np.arange(1, int((m - 1) / 2 + 1)), group_size)  # first part
            clusters1 = clusters1[:, None]  # reshape array
            clusters2 = np.repeat(int((m / 2 + 1)), middle_size + 1)  # mid part
            clusters2 = clusters2[:, None]
            clusters3 = np.repeat(np.arange(int((m - 1) / 2 + 2), m + 1), group_size)  # last part
            clusters3 = clusters3[:, None]
            cluster_array = np.concatenate(
                (clusters1, clusters2, clusters3), axis=None)  # put together the 3 arrays
            cluster_array = cluster_array[:, None]
            final_array = np.append(
                df_array, cluster_array, axis=1)  # final array with assigned clusters
            df_final = pd.DataFrame(final_array)  # final dataframe with assigned clusters
            df_final.columns = [*df_final.columns[:-1], 'Cluster']  # rename only last column
        return df_final

    # function to get the minimum for each column
    def min(self, data):  # input any dataframe with integers
        min_col = {}
        for i in data:
            min_col[i] = data[i].min()
            result = pd.DataFrame([min_col], index=['min'])
        return result  # returns min for each column

    # function to calculate mean for columns and add noise
    def mean_calc(
        self,
        distance_df: pd.DataFrame,
        group_size: int,
        epsilon: float
    ) -> pd.DataFrame:
        row_id = distance_df.iloc[:, 0]  # pass row id column to hold
        distance_df.drop(distance_df.columns[0], axis=1, inplace=True)
        dist = distance_df.iloc[:, 0]  # pass distance column to hold
        distance_df.drop(distance_df.columns[0], axis=1, inplace=True)
        lbl = distance_df.iloc[:, len(distance_df.columns) - 2]  # pass label column to hold
        distance_df.drop(
            distance_df.columns[len(distance_df.columns) - 2], axis=1,
            inplace=True
        )  # drop label column from dataframe
        clust = distance_df.iloc[:, -1]  # pass cluster column to hold
        mean_df = distance_df.groupby(
            ['Cluster']).transform('mean')  # transform each cluster with the mean

        # # transform each cluster with peak to peak, redundant?
        # ptp = distance_df.groupby(
        #     ['Cluster']).transform(np.ptp)
        # delta = (ptp / group_size)
        # compute noise for each cluster
        # mean_array = distance_df.groupby(['Cluster']).agg('mean').to_numpy()
        delta_array = distance_df.groupby(
            ['Cluster']).agg(np.ptp).to_numpy()  # transform each cluster with peak to peak
        count_array = distance_df.groupby(['Cluster']).size().to_numpy()

        num_row = distance_df.shape[0]
        num_cols = delta_array.shape[1]
        num_group = np.floor(num_row / group_size).astype(int)
        laplace_df = np.empty((0, num_cols), float)  # aplace hodler for laplace noise

        for i in range(num_group):
            # compute delta
            # count of cluster i
            delta_local = delta_array[i, :] / count_array[i]
            laplace = np.random.laplace(0, delta_local / epsilon)
            laplace = np.reshape(laplace, (-1, num_cols))
            laplace = np.repeat(laplace, count_array[i], axis=0)
            laplace_df = np.append(laplace_df, laplace, axis=0)

        laplace_df = pd.DataFrame(laplace_df)
        laplace_df.shape
        mean_df.shape
        meanlap_df = pd.DataFrame(
            mean_df.values + laplace_df.values,
            columns=mean_df.columns
        )  # adding mean and laplace dataframe
        meanlap_df.insert(
            loc=0, column='row_id', value=row_id)  # insert column back(front of dataframe)
        meanlap_df.insert(
            loc=1, column='distance', value=dist)  # insert column back(front of dataframe)
        meanlap_df.insert(
            loc=len(meanlap_df.columns),
            column='Cluster',
            value=clust)  # insert column back(end of dataframe)
        meanlap_df.insert(
            loc=len(meanlap_df.columns) - 1,
            column='label',
            value=lbl)  # insert column back(end of dataframe)
        return meanlap_df  # returns laplace dataframe


def main():
    file_name = args.filename

    dp = DataProcess()
    ipv4 = IPv4()
    ipv6 = IPv6()

    if not dp.is_validate_csv_file(file_name):
        print('File should be a CSV file')
        sys.exit()

    eps = 1
    num_instances = 20
    # wt = 0
    # ite = 5
    sep = ','
    priv = True
    acc = False
    header_exists = True
    col_list = (
        '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 '
        '29 30 31 32 33 34 35 36 37 38 39 40 41')
    label_col_id = '42'
    use_context = False

    df = dp.read_dataset(file_name, sep, header_exists)
    col1 = df.iloc[:, 0]
    ip1 = col1[0]
    df1 = None

    global ip_version

    if ipv4.validate_ipv4(ip1):
        df1 = ipv4.call_ipv4_ops(df, int(num_instances))
        ip_version = 4
    elif ipv6.validate_ipv6(ip1):
        ip_version = 6
        df1 = ipv6.call_ipv6_ops(df, int(num_instances))

    df2, anon_fields = dp.create_df_to_work_on(df1, col_list, label_col_id, use_context)

    dp.condense_perclass(
        df2,
        df,
        float(eps),
        int(num_instances),
        priv,
        acc,
        label_col_id,
        anon_fields
    )
    print("successfully completed!!!")


# command python app.py --file <your finename>
if __name__ == "__main__":
    main()
