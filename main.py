from collections import Counter

import numpy as np
import math
import decimal
import locale
import re
import random
import socket
from sklearn.metrics import mean_squared_error
from math import sqrt


import secrets

# modified condensation D Code
import pandas as pd
# import pylab as pl
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from scipy.spatial import distance
import cProfile as profile
from sklearn import preprocessing
import pstats
# IPv4 lists and variables
IPADDRESS_PATTERN_IPv4 = '^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'

ipClusterIPv4 = {}
ipNewReplaceorIPv4 = {}
ipLastSumIPv4 = {}
sortedNewIPv4 = []
sortedNewIpMapIPv4 = {}
newIpCountMapIPv4 = {}

# IPv6 lists and variables
IPv6_PATTERN = '(?:^|(?<=\s))(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))(?=\s|$)'
MULTICAST_PATTERN = '^(FF00|ff00|Ff00|fF00)'

UNICAST_IPv6 = 1
MULTICAST_IPv6 = 2
cluster_id_IPv6 = 1
IP_version = None
cut_threshold = 0.001
source_IPs = []  # keeps the list of the IP before anonymization
anon_list_IPv6 = []  # keeps the list of the network prefix of the IPv6 (to be anonymized)
# preserve_list_IPv6 = [] #keeps the list of the host suffix of the IPv6 (to be averaged to create groups/clusters of K)
ipPrefixCountIPv6 = {}  # a map, stores the number of times a network prefix appears within the dataset
ipPrefixReplaceorIPv6 = {}  # a map, stores the anoymized relacement of a network prefix
ipFirstSumIPv6 = {}  # a map, stores the sum of the first part of the host suffix against its prefix
ipSecondSumIPv6 = {}  # a map, stores the sum of the second part of the host suffix against its prefix
ipThirdSumIPv6 = {}  # a map, stores the sum of the third part of the host suffix against its prefix
ipFourthSumIPv6 = {}  # a map, stores the sum of the fourth part of the host suffix against its prefix
ipFifthSumIPv6 = {}  # a map, stores the sum of the fifth part of the host suffix against its prefix
clusterIDMapIPv6 = {}  # a map, stores the cluster IP of an IP against its prefix
# new maps and lists to do clustering (group by)
prefixMapsAnonIPv6 = {}  # a map, stores the anoymized IPv6 against its network prefix
IPv6_types = {}  # a map, stores whether an IPv6 is unicast or multicast
unclusteredIPsIPv6 = []  # it stores the list of the IPs, that are not in any cluster yet


def readDataset(filename, seperator, headerExists):
    df1 = pd.DataFrame()
    if (headerExists == "n" or headerExists == "N"):
        df1 = pd.read_csv(filename, sep=seperator, header=None)
    else:
        df1 = pd.read_csv(filename, sep=seperator)
    if (len(df1.columns) == 1):
        df1 = pd.read_csv(filename, sep="\t", header=None)
    # print(df1)
    return df1


""" IPv6 anonymization"""


def validateIPv6(ipAdr):
    Regex = re.compile(IPv6_PATTERN)
    return (Regex.search(str(ipAdr)) != None)


# initializes variables for reusing
def initializeVariables():
    global anon_list_IPv6, ipPrefixCountIPv6, ipPrefixReplaceorIPv6, ipFirstSumIPv6, ipSecondSumIPv6, ipThirdSumIPv6, ipFourthSumIPv6, \
        ipFifthSumIPv6, clusterIDMapIPv6, prefixMapsAnonIPv6, prefixMapsAnonIPv6, IPv6_types, unclusteredIPsIPv6
    anon_list_IPv6 = []  # keeps the list of the network prefix of the IPv6 (to be anonymized)
    # preserve_list_IPv6 = [] #keeps the list of the host suffix of the IPv6 (to be averaged to create groups/clusters of K)

    ipPrefixCountIPv6 = {}  # a map, stores the number of times a network prefix appears within the dataset
    ipPrefixReplaceorIPv6 = {}  # a map, stores the anoymized relacement of a network prefix
    ipFirstSumIPv6 = {}  # a map, stores the sum of the first part of the host suffix against its prefix
    ipSecondSumIPv6 = {}  # a map, stores the sum of the second part of the host suffix against its prefix
    ipThirdSumIPv6 = {}  # a map, stores the sum of the third part of the host suffix against its prefix
    ipFourthSumIPv6 = {}  # a map, stores the sum of the fourth part of the host suffix against its prefix
    ipFifthSumIPv6 = {}  # a map, stores the sum of the fifth part of the host suffix against its prefix
    clusterIDMapIPv6 = {}  # a map, stores the cluster IP of an IP against its prefix

    # new maps and lists to do clustering (group by)
    prefixMapsAnonIPv6 = {}  # a map, stores the anoymized IPv6 against its network prefix
    IPv6_types = {}  # a map, stores whether an IPv6 is unicast or multicast
    unclusteredIPsIPv6 = []  # it stores the list of the IPs, that are not in any cluster yet


"""The following function returns whether a given ip address is unicast or multicast. 
It also returns how many parts of that IPv6 needs to be anonymized, and how many parts should 
be used to create cluster/groups. For example, let a:b:c:d:e:f:g:h an IPv6 address.
 If it is unicast, then the first 3 parts need to be anonymized (a:b:c), and the last 5 parts 
 should be averaged with the same cluster IPs. This averaging will be by parts, meaning 'd's 
 will be averaged separately, 'e's will be averaged separately and so on. 

If it is a multicast IPv6, then first 4 parts ( a:b:c:d) needs to be anonymized.
"""


def IPv6_type_detection(ipAdr):
    if (re.search(MULTICAST_PATTERN, ipAdr) == None):
        return UNICAST_IPv6, 3
    else:
        return MULTICAST_IPv6, 4


"""The following function creates a complete IPv6 string from the input IPv6. 
If there are consecutive 0's in the IPv6 string (ex: 0:0:0), usually it's written as (::). 
This format can create trouble in the later phase of the anonymization, as different IPs may
 have different length/format. To solve that, this function replaces any '::' with necessary 
 number of 0's."""


def create_ip_string_IPv6(ipAddr):
    parts = ipAddr.split(':')
    num_parts = len(parts)
    new_IPv6 = ''
    zero_parts = 8 - num_parts + 1

    for i in range(0, num_parts):
        if (len(parts[i]) == 0):
            for j in range(0, zero_parts):
                new_IPv6 += '0:'
        else:
            new_IPv6 += (parts[i] + ':')

    new_IPv6 = new_IPv6[0: len(new_IPv6) - 1]
    return new_IPv6


"""This function seperates the part of ip address that needs to be anonymized, and that 
will be averaged for clustering."""


def separate_ip_parts_IPv6(ipAddr, anon_bit):
    non_anon_str = ''
    anon_str = ''
    parts = ipAddr.split(':')

    # separates the portion to be anonymized
    for i in range(0, anon_bit):
        anon_str += (parts[i] + ':')
    anon_str = anon_str[0: len(anon_str) - 1]

    # separates the portion to be averaged
    for i in range(anon_bit, 8):
        non_anon_str += (parts[i] + ':')
    non_anon_str = non_anon_str[0: len(non_anon_str) - 1]

    return non_anon_str, anon_str


"""Returns random hexadecimal number. It's needed for random anonymizing the network prefix."""


def randomHexNum():
    return secrets.token_hex(2)


"""This function updates map to hold the sum of host parts of IPs that are in the same cluster. 
The input 'map' holds information about which part the mathematical opertion is going on."""


def createSuffixReplacorIPv6(map, key, new_val):
    if map.get(key) is not None:
        update_val = int(map.get(key), 16) + int(new_val, 16)
        map.update({key: hex(update_val)[2:]})
    else:
        map.update({key: new_val})


"""This function updates different data structures. It also does the random permutation of network 
prefix numbers."""


def updateIPv6Maps(PreservedIpAddr, AnonIpAddr, anonParts):
    # updates the number of instances of a prefix
    if ipPrefixCountIPv6.get(AnonIpAddr) is not None:
        ipPrefixCountIPv6.update({AnonIpAddr: ipPrefixCountIPv6.get(AnonIpAddr) + 1})
    else:
        ipPrefixCountIPv6.update({AnonIpAddr: 1})

    pre = ''

    # ipPrefixReplaceor holds the permuted ip addresses
    if ipPrefixReplaceorIPv6.get(AnonIpAddr) is not None:
        pre = ipPrefixReplaceorIPv6.get(AnonIpAddr)
    else:
        parts = AnonIpAddr.split(':')
        # it's better to not change the first part, as it storess info about unicast or  multicast
        pre = parts[0]
        # replaces each part of the anonymizable IP with a random hexadecimal number
        for i in range(1, anonParts):
            pre += (':' + str(randomHexNum()))

        ipPrefixReplaceorIPv6.update({AnonIpAddr: pre})

    nonAnon = PreservedIpAddr.split(':')
    # print(preserved)
    # update the maps that store the sum of the suffix parts
    createSuffixReplacorIPv6(ipFirstSumIPv6, AnonIpAddr, nonAnon[0])
    createSuffixReplacorIPv6(ipSecondSumIPv6, AnonIpAddr, nonAnon[1])
    createSuffixReplacorIPv6(ipThirdSumIPv6, AnonIpAddr, nonAnon[2])
    createSuffixReplacorIPv6(ipFourthSumIPv6, AnonIpAddr, nonAnon[3])

    if anonParts == 3:  # ip type is unicast
        createSuffixReplacorIPv6(ipFifthSumIPv6, AnonIpAddr, nonAnon[4])


"""This function computes the mean of each part of the suffix with the same prefix."""


def computeMeanofKIPv6(AnonIpAddr, ip_type, num):
    suffix = ''

    sums = ipFirstSumIPv6.get(AnonIpAddr)
    mean = hex(int(sums, 16) // num)[2:]
    suffix += (':' + str(mean))

    sums = ipSecondSumIPv6.get(AnonIpAddr)
    mean = hex(int(sums, 16) // num)[2:]
    suffix += (':' + str(mean))

    sums = ipThirdSumIPv6.get(AnonIpAddr)
    mean = hex(int(sums, 16) // num)[2:]
    suffix += (':' + str(mean))

    sums = ipFourthSumIPv6.get(AnonIpAddr)
    mean = hex(int(sums, 16) // num)[2:]
    suffix += (':' + str(mean))

    if ip_type == UNICAST_IPv6:
        sums = ipFifthSumIPv6.get(AnonIpAddr)
        mean = hex(int(sums, 16) // num)[2:]
        suffix += (':' + str(mean))

    return suffix


"""This function computes the mean of the suffix part with different prefix. The function is called 
for each part of the suffix separately."""


def computeMeanMultipleIPv6(listOfIPs, map):
    l = len(listOfIPs)
    first = listOfIPs[0]
    num = ipPrefixCountIPv6.get(first)
    sums = int(map.get(first), 16)
    for i in range(1, l):
        ip = listOfIPs[i]
        num += ipPrefixCountIPv6.get(ip)
        sums += int(map.get(ip), 16)

    mean = hex(sums // num)[2:]

    return str(mean)


"""This replaces the ip addresses with anonymized ips."""


def replaceorIPv6(AnonIpAddr, K):
    prefix = ipPrefixReplaceorIPv6.get(AnonIpAddr)
    num = ipPrefixCountIPv6.get(AnonIpAddr)
    IPv6_type = IPv6_types.get(AnonIpAddr)
    suffix = ''
    new_IP = ''
    global cluster_id_IPv6

    # print('cluster ',cluster_id_IPv6)
    # print(AnonIpAddr, ' ', num)
    unclusteredIPsIPv6.remove(AnonIpAddr)

    if num >= K:  # if the prefix has equal to or more than K instances
        # print('inside if')
        suffix = computeMeanofKIPv6(AnonIpAddr, IPv6_type, num)
        new_IP = prefix + suffix
        prefixMapsAnonIPv6.update({prefix: new_IP})
        clusterIDMapIPv6.update({prefix: cluster_id_IPv6})

    else:  # if multiple IPs need to be clustered
        # print('inside else')
        listToMakeK = []  # holds the list of the prefixes that help make a cluster of size K
        listlen = len(unclusteredIPsIPv6)

        # print(IPv6_type)
        for i in range(0, listlen):
            ip = unclusteredIPsIPv6[i]
            if IPv6_type == IPv6_types.get(ip):
                listToMakeK.append(ip)
                num += ipPrefixCountIPv6.get(ip)

            # the following conditional makes sure that only one group with less than k
            # instances is not left out
            if (i == listlen - 2):
                ip1 = list(unclusteredIPsIPv6)[listlen - 1]
                c = ipPrefixCountIPv6.get(ip1)
                if (c < K and IPv6_type == IPv6_types.get(ip1)):
                    num += c
                    listToMakeK.append(ip1)
                    break

            if (num >= K):
                break

        # print(len(listToMakeK))
        # print(len(unclusteredIPs))
        # print(unclusteredIPs)

        # remove the IPs that have been anonymized from the list, so that they are not calculated again
        for i in range(0, len(listToMakeK)):
            # print(listToMakeK[i])
            unclusteredIPsIPv6.remove(listToMakeK[i])
            # print('removed')

        listToMakeK.append(AnonIpAddr)

        suffix += (':' + computeMeanMultipleIPv6(listToMakeK, ipFirstSumIPv6))
        suffix += (':' + computeMeanMultipleIPv6(listToMakeK, ipSecondSumIPv6))
        suffix += (':' + computeMeanMultipleIPv6(listToMakeK, ipThirdSumIPv6))
        suffix += (':' + computeMeanMultipleIPv6(listToMakeK, ipFourthSumIPv6))
        if IPv6_type == UNICAST_IPv6:
            suffix += (':' + computeMeanMultipleIPv6(listToMakeK, ipFifthSumIPv6))

        new_IP = prefix + suffix

        for i in range(0, len(listToMakeK)):
            pre = ipPrefixReplaceorIPv6.get(listToMakeK[i])
            prefixMapsAnonIPv6.update({pre: (pre + suffix)})
            clusterIDMapIPv6.update({pre: cluster_id_IPv6})

    cluster_id_IPv6 += 1
    # print('done')
    return new_IP


"""main function starts here. Reads the dataset, validate IPs and update maps."""


def SeparateIPv6(test_ip):
    IPv6_type = None
    anon_parts = 0
    non_anonymizable_ip = None
    anonymizable_ip = None

    IPv6_type, anon_parts = IPv6_type_detection(test_ip)

    # print(IPv6_type)
    test_ip = create_ip_string_IPv6(test_ip)
    non_anonymizable_ip, anonymizable_ip = separate_ip_parts_IPv6(test_ip, anon_parts)
    anon_list_IPv6.append(anonymizable_ip)
    # IP_version.update({anonymizable_ip: 'IPv6'})
    IPv6_types.update({anonymizable_ip: IPv6_type})

    updateIPv6Maps(non_anonymizable_ip, anonymizable_ip, anon_parts)


"""Anonymize the IPs."""


def createAnonIPv6(numInstances):
    temp_dict = sorted(ipPrefixCountIPv6.items(), key=lambda item: item[1])
    for i in temp_dict:
        # print(i[0], ' ', i[1])
        unclusteredIPsIPv6.append(i[0])

    # print(unclusteredIPs)
    unclusteredIPs_copy = unclusteredIPsIPv6.copy()
    l = len(unclusteredIPs_copy)
    for i in range(0, l):
        ip = unclusteredIPs_copy[i]
        if ip in unclusteredIPsIPv6:
            newIP = replaceorIPv6(ip, numInstances)


def getNonBlankInput(message, error_message):
    x = input(message)
    while len(x.strip()) == 0:
        x = input(error_message)

    return x


def callIPv6Ops(df, numInstances):
    # print(prefixMapsAnonIPv6)
    initializeVariables()
    # print(prefixMapsAnonIPv6)
    source_IPs = df.iloc[:, 0]
    dest_IPs = []
    # print(source_IPs)
    for i in range(0, len(source_IPs)):
        test_ip = source_IPs[i]
        if (validateIPv6(test_ip)):
            SeparateIPv6(test_ip)

    createAnonIPv6(numInstances)

    # source_ip_anonymized = []
    # dest_ip_anonymized = []
    # print(len(anon_list_IPv6))
    for i in range(0, len(anon_list_IPv6)):
        prefix = ipPrefixReplaceorIPv6.get(anon_list_IPv6[i])
        nip = prefixMapsAnonIPv6.get(prefix)
        # source_ip_anonymized.append(nip)
        # print(nip)
        df.iloc[i, 0] = nip
    # print(df)

    if (len(df.columns) > 1):  # call for destination ips
        # print('multiple columns')
        initializeVariables()
        dest_IPs = df.iloc[:, 1]

        for i in range(0, len(dest_IPs)):
            test_ip = dest_IPs[i]
            if (validateIPv6(test_ip)):
                SeparateIPv6(test_ip)

        createAnonIPv6(numInstances)

        # print(len(anon_list_IPv6))
        for i in range(0, len(anon_list_IPv6)):
            prefix = ipPrefixReplaceorIPv6.get(anon_list_IPv6[i])
            nip = prefixMapsAnonIPv6.get(prefix)
            # dest_ip_anonymized.append(nip)
            df.iloc[i, 1] = nip

    return df


""" IPv6 anonymization ends"""

"""IPv4 anonymization"""


def validateIPv4(ipAdr):
    # print('In Validation for:' + str(ipAdr))
    haRegex = re.compile(IPADDRESS_PATTERN_IPv4)
    return haRegex.search(str(ipAdr)) != None


def randomNum(high, low):
    return random.randint(0, high - low) + low


def updateMapsIPv4(updateDf):
    # df1 = pd.read_csv(inputfile, sep="\t")
    src_ip = updateDf.iloc[:, 0]  # df1['SRC_IP']
    dest_ip = updateDf.iloc[:, 1]
    for i in range(len(updateDf) - 1):
        # print(str(i))

        if (validateIPv4(src_ip[i])):
            # print('Validated Source in UpdateMaps: ' + str(i) + ':' + src_ip[i])
            ip = src_ip[i].split('.')
            last = int(ip[len(ip) - 1])
            key = ''
            for j in range(len(ip) - 1):
                key = key + ip[j] + "."

            if ipClusterIPv4.get(key) is not None:
                ipClusterIPv4.update({key: ipClusterIPv4.get(key) + 1})
            else:
                ipClusterIPv4.update({key: 1})
            pre = ''

            if ipNewReplaceorIPv4.get(key) is not None:
                pre = ipNewReplaceorIPv4.get(key)
            else:
                pre = str(random.randint(0, 254) + 1) + "." + str(random.randint(0, 254) + 1) + "." + str(
                    random.randint(0, 254) + 1) + "."
                ipNewReplaceorIPv4.update({key: pre})

            if ipLastSumIPv4.get(key) is not None:
                ipLastSumIPv4.update({key: int(ipLastSumIPv4.get(key)) + last})
            else:
                ipLastSumIPv4.update({key: last})

        if (validateIPv4(dest_ip[i])):
            # print('Validated Destination in UpdateMaps: ' + str(i) + ':' + dest_ip[i])
            ip = dest_ip[i].split('.')
            last = int(ip[len(ip) - 1])
            key = ''
            for j in range(len(ip) - 1):
                key = key + ip[j] + "."

            if ipClusterIPv4.get(key) is not None:
                # print('found ipCluster key:' + key)
                ipClusterIPv4.update({key: ipClusterIPv4.get(key) + 1})
            else:
                ipClusterIPv4.update({key: 1})
            pre = ''

            if ipNewReplaceorIPv4.get(key) is not None:
                # print('found ipNewReplaceor key:' + key)
                pre = ipNewReplaceorIPv4.get(key)
            else:
                pre = str(randomNum(210, 10)) + "." + str(randomNum(210, 10)) + "." + str(randomNum(210, 10)) + "."
                ipNewReplaceorIPv4.update({key: pre})

            if ipLastSumIPv4.get(key) is not None:
                # print('found ipLastSum key:' + key)
                ipLastSumIPv4.update({key: int(ipLastSumIPv4.get(key)) + last})
            else:
                ipLastSumIPv4.update({key: last})
    # print('ipNewReplaceor')
    # print(ipNewReplaceor)
    # print('ipCluster')
    # print(ipCluster)
    # print('ipLastSum')
    # print(ipLastSum)


def replaceorIPv4(replaceDf):  # , resultFile):
    # df1 = pd.read_csv(inputFile, sep="\t")
    src_ip = replaceDf.iloc[:, 0]
    dest_ip = replaceDf.iloc[:, 1]
    for i in range(len(replaceDf)):
        # print('i:' + str(i))
        newSrc = ''
        newDest = ''
        if (validateIPv4(src_ip[i])):
            # print('Source Validated')
            ip = src_ip[i].split('.')
            # print(src_ip[i])
            key = ''
            for j in range(len(ip) - 1):
                key = key + ip[j] + "."
            # print('Source key:' + str(key))
            pre = ipNewReplaceorIPv4.get(key)
            # print('pre' + str(pre))
            num = ipClusterIPv4.get(key)
            # print('num' + str(num))
            sums = ipLastSumIPv4.get(key)
            # print('sums' + str(sums))
            mean = sums // num
            newSrc = str(pre) + str(mean)
        else:
            newSrc = src_ip[i]

        if (validateIPv4(dest_ip[i])):
            # print('destination Validated' + dest_ip[i])
            ip = dest_ip[i].split('.')
            key = ''
            for j in range(len(ip) - 1):
                key = key + ip[j] + "."
            # print('Dest key:' + str(key))
            pre = ipNewReplaceorIPv4.get(key)
            # print('pre' + str(pre))
            num = ipClusterIPv4.get(key)
            # print('num' + str(num))
            sums = ipLastSumIPv4.get(key)
            # print('sums' + str(sums))
            mean = sums // num
            newDest = str(pre) + str(mean)
        else:
            newDest = dest_ip[i]

        replaceDf.iloc[i, 0] = newSrc
        # print('new src ip:' + str(newSrc))
        # print('SrcIp after Replacing:')
        # print(replaceDf.iloc[i, 0])
        replaceDf.iloc[i, 1] = newDest
    print(replaceDf)
    return replaceDf


def getInfoIPv4(dfInfo):
    # df1 = pd.read_csv(resultFile, sep="\t")
    src_ip = dfInfo.iloc[:, 0]  # df1['SRC_IP']
    dest_ip = dfInfo.iloc[:, 1]
    for i in range(len(dfInfo) - 1):
        if (validateIPv4(src_ip[i])):
            if src_ip[i] in newIpCountMapIPv4:
                newIpCountMapIPv4.update({src_ip[i]: newIpCountMapIPv4.get(src_ip[i]) + 1})
            else:
                newIpCountMapIPv4.update({src_ip[i]: 1})
                sortedNewIPv4.append(src_ip[i])

        if (validateIPv4(dest_ip[i])):
            if dest_ip[i] in newIpCountMapIPv4:
                newIpCountMapIPv4.update({dest_ip[i]: newIpCountMapIPv4.get(dest_ip[i]) + 1})
            else:
                newIpCountMapIPv4.update({dest_ip[i]: 1})
                sortedNewIPv4.append(dest_ip[i])

        sortedNewIPv4.sort()
        for j in range(len(sortedNewIPv4)):
            sortedNewIpMapIPv4.update({sortedNewIPv4[j]: j})


def getNearIPv4(ip, k):
    result = ip
    index = sortedNewIpMapIPv4.get(ip)
    count = newIpCountMapIPv4.get(ip)

    if (count < k):
        dis = 0;
        for i in range(len(sortedNewIPv4)):
            if (newIpCountMapIPv4.get(sortedNewIPv4[i]) >= k):
                dis = i - index
                break
        for i in range(index - 1, 0):
            if (newIpCountMapIPv4.get(sortedNewIPv4[i]) >= k):
                if ((index - i) < dis):
                    dis = i - index
                    break
        newIndex = index + dis
        result = sortedNewIPv4[newIndex]
    return result


def aggregationIPv4(dfReplacedArg, numInstances):  # , newFile):
    # df1 = pd.read_csv(resultFile, sep="\t")
    #src_ip = dfReplacedArg.iloc[:, 0]  # df1['SRC_IP']
    src_ip = dfReplacedArg.iloc[:, 0].to_numpy()
    #dest_ip = dfReplacedArg.iloc[:, 1]
    dest_ip = dfReplacedArg.iloc[:, 1].to_numpy()
    for i in range(len(dfReplacedArg) - 1):
        #if (validateIPv4(src_ip[i])):
        #    dfReplacedArg.iloc[i, 0] = getNearIPv4(src_ip[i], numInstances)
        if (validateIPv4(src_ip[i])):
            src_ip[i] = getNearIPv4(src_ip[i], numInstances)
        #if (validateIPv4(dest_ip[i])):
        #    dfReplacedArg.iloc[i, 1] = getNearIPv4(dest_ip[i], numInstances)
        if (validateIPv4(dest_ip[i])):
            dest_ip[i] = getNearIPv4(dest_ip[i], numInstances)
    dfReplacedArg.iloc[:,0] = src_ip
    dfReplacedArg.iloc[:, 1] = dest_ip
    # df1.to_csv(newFile, sep=',')
    return dfReplacedArg


# IP Replacer Code Ends

"""IPv4 anonymization ends"""


def euclid_weighted(v1, v2, wt, numattr):
    diffv = abs((v1 - v2))
    sums = 0
    wt1 = (1 - wt) / (numattr - 1)
    for j in range(numattr - 1):
        sums = sums + (wt1 * (diffv[j + 1] ** 2))
    sums = sums + (wt * (diffv[numattr] ** 2))
    return math.sqrt(sums)


# In[46]:


def privacy(data_file, modified_file, output_file, conf):
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
    #data = pd.read_csv(data_file, sep="\t")
    data = pd.read_csv(data_file, sep=",",header=None).to_numpy()
    #nrows = data.shape[0]
    nrows, ncols = data.shape
    #ncols = data.shape[1]

    #modified_data = pd.read_csv(modified_file, sep="\t")
    #Diffdata = pd.DataFrame(np.zeros((nrows, ncols)))
    modified_data = pd.read_csv(modified_file, sep=",", header=None).to_numpy()
    #diffdata = pd.DataFrame(np.zeros((nrows, ncols))).to_numpy()
    diffdata = modified_data - data
    diffdata_norm = np.copy(diffdata)
    #   compute relative difference between perturbed and original, not a standard metric but give some idea
    #max_array = np.maximum(modified_data,data)
    #set zero entry to 1
    #max_array[max_array==0]=1
    #relative_diff = np.abs(np.subtract(modified_data,data)/max_array)
    #mean_relative_diff = np.mean(relative_diff,axis=0)
    #mean_relative_diff_val=np.mean(mean_relative_diff)
    #median_relative_diff_val=np.median(mean_relative_diff)
    #print("Average relative difference:" + str(mean_relative_diff_val))
    #print("Median relative difference:" + str(median_relative_diff_val))
    #    for i in range(nrows):
#        diffdata.values[i, :] = modified_data.values[i, :].astype(float) - data.values[i, :].astype(float)
    ranges = []
    nrsme = np.zeros(ncols)
    nrsme2 = np.zeros(ncols)
    for i in range(0, ncols):
        ranges.append(float(np.max(data[:, i])) - float(np.min(data[:, i])))
        #print(ranges)

        if ranges[i] > 0:
 #           diffdata.values[:, i] = diffdata.values[:, i].astype(int) / int(ranges[i])
 #          how about normalize by standard deviation
            #diffdata[:, i] = diffdata[:, i] / ranges[i]
            stdi2 = np.std(data[:,i])
            stdi = np.ptp(data[:,i])
            diffdata[:, i] = diffdata[:, i] / stdi
            diffdata_norm[:,i] = diffdata_norm[:, i] / stdi2
            nrsme[i] =  sqrt(mean_squared_error(data[:,i],modified_data[:,i]))/stdi
            nrsme2[i] = sqrt(mean_squared_error(data[:, i], modified_data[:, i])) / stdi2
#            data[:, i] = data[:, i] / ranges[i]
    #diffdata.to_csv('diffdata.txt', sep=',', index=False)
    nrsme_mean = np.mean(nrsme)
    nrsme2_mean = np.mean(nrsme2)
    print("root mean squared error normalized min-max:" + str(nrsme_mean))
    print("root mean squared error normalized std:" + str(nrsme2_mean))
    upper = (1 + conf) / 2.0
    lower = (1 - conf) / 2.0
    privacy = np.zeros((4, ncols))
#    privacy[0, :] = diffdata.quantile(upper) - diffdata.quantile(lower)
    privacy[0, :] = np.quantile(diffdata,upper,axis=0) - np.quantile(diffdata,lower,axis=0)
    privacy[3, :] = np.quantile(diffdata_norm, upper, axis=0) - np.quantile(diffdata_norm, lower, axis=0)
    #varDiffData = diffdata.var(axis=0)
    #varData = data.var(axis=0)
    varDiffData = np.var(diffdata,axis=0)
    varData = np.var(data,axis=0)
    for y in range(len(varDiffData)):
        if varData[y] == 0:
            varData[y] = 1
        privacy[1, y] = varDiffData[y] / varData[y]

    #diffdata2 = diffdata.abs()
    diffdata2 = np.abs(diffdata)
    #diffdata2.to_csv('diff.txt', sep=',')
    #privacy[2, :] = diffdata2.quantile(0.5)
    privacy[2, :] = np.quantile(diffdata2,0.5,axis=0)
    #print(privacy)
    privacy_mean = privacy.mean(axis=1)
    print(privacy_mean)
    meanPrivacy = pd.DataFrame(privacy_mean)
    meanPrivacy.to_csv(output_file, sep=',')
    r = privacy_mean[0]
    privacy_val = privacy_mean[0]
    privacy_val2 = privacy_mean[3]
    print("Privacy:" + str(privacy_val))
    print("Privacy normalized by std:" + str(privacy_val2))
#   compute relative difference between perturbed and original, not a standard metric but give some idea


##modified condensation p starts here##


# function to get the minimum for each column
def min(data):  # input any dataframe with integers
    min_col = {}
    for i in data:
        min_col[i] = data[i].min()
        result = pd.DataFrame([min_col], index=['min'])
    return result  # returns min for each column


# function to create clustering
def clustering(sorted_df, group_size):  # input will be dataframe with distances & indexes
    rows1 = sorted_df.index
    rows = len(rows1)
    m = rows / group_size  # calculate number of clusters
    m = math.floor(m)
    m = int(m)
    if (m % 2) == 0:
        middle_size = (rows - (m / 2 - 1) * group_size) - ((m / 2) * group_size + 1)  # calculate size of mid cluster
        df_array = sorted_df.to_numpy()  # dataframe to array
        clusters1 = np.repeat(np.arange(1, int(m / 2 + 1)), group_size)  # first part
        clusters1 = clusters1[:, None]  # reshape array
        clusters2 = np.repeat(int((m / 2 + 1)), middle_size + 1)  # mid part
        clusters2 = clusters2[:, None]
        clusters3 = np.repeat(np.arange(int((m / 2) + 2), m + 1), group_size)  # last part
        clusters3 = clusters3[:, None]
        cluster_array = np.concatenate((clusters1, clusters2, clusters3), axis=None)  # put together the 3 arrays
        cluster_array = cluster_array[:, None]
        final_array = np.append(df_array, cluster_array, axis=1)  # final array with assigned clusters
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
        cluster_array = np.concatenate((clusters1, clusters2, clusters3), axis=None)  # put together the 3 arrays
        cluster_array = cluster_array[:, None]
        final_array = np.append(df_array, cluster_array, axis=1)  # final array with assigned clusters
        df_final = pd.DataFrame(final_array)  # final dataframe with assigned clusters
        df_final.columns = [*df_final.columns[:-1], 'Cluster']  # rename only last column
    return df_final


# function to calculate mean for columns and add noise
def mean_calc(distance_df, group_size, epsilon):  # input distances dataframe with assigned cluster
    row_id = distance_df.iloc[:, 0]  # pass row id column to hold
    distance_df.drop(distance_df.columns[0], axis=1, inplace=True)  # drop row id column from dataframe
    dist = distance_df.iloc[:, 0]  # pass distance column to hold
    distance_df.drop(distance_df.columns[0], axis=1, inplace=True)  # drop distance column from dataframe
    lbl = distance_df.iloc[:, len(distance_df.columns) - 2]  # pass label column to hold
    distance_df.drop(distance_df.columns[len(distance_df.columns) - 2], axis=1,
                     inplace=True)  # drop label column from dataframe
    clust = distance_df.iloc[:, -1]  # pass cluster column to hold
    mean_df = distance_df.groupby(['Cluster']).transform('mean')  # transform each cluster with the mean
    ptp = distance_df.groupby(['Cluster']).transform(np.ptp)  # transform each cluster with peak to peak, redundant?
    delta = (ptp / group_size)
    # compute noise for each cluster
    mean_array = distance_df.groupby(['Cluster']).agg('mean').to_numpy()
    delta_array = distance_df.groupby(['Cluster']).agg(np.ptp).to_numpy()# transform each cluster with peak to peak
    count_array = distance_df.groupby(['Cluster']).size().to_numpy()
    #delta_array = delta.to_numpy()  # delta to array to add laplace noise
    num_row = distance_df.shape[0]
    num_cols = delta_array.shape[1]
    num_group = np.floor(num_row / group_size).astype(int)
    laplace_df = np.empty((0,num_cols), float) # aplace hodler for laplace noise
    #laplace = np.random.laplace(0, delta_array / epsilon)
    for i in range(num_group):
      # compute delta
      # count of cluster i
      delta_local = delta_array[i,:]/count_array[i] # compute local delta within the cluster
      laplace =  np.random.laplace(0, delta_local / epsilon) # laplace noise for each column
      laplace = np.reshape(laplace,(-1,num_cols)) # reshape it to 2d array
      laplace = np.repeat(laplace,count_array[i], axis=0) # same noise for all rows in the same cluster
      #laplace = np.reshape(laplace,(count_array[i],num_cols))
      laplace_df = np.append(laplace_df,laplace,axis=0) # append to existing noise rows
    #laplace = delta_array / epsilon * np.random.laplace(0, 1, 1)



    laplace_df = pd.DataFrame(laplace_df)
    laplace_df.shape
    mean_df.shape
    meanlap_df = pd.DataFrame(mean_df.values + laplace_df.values,
                              columns=mean_df.columns)  # adding mean and laplace dataframe
    meanlap_df.insert(loc=0, column='row_id', value=row_id)  # insert column back(front of dataframe)
    meanlap_df.insert(loc=1, column='distance', value=dist)  # insert column back(front of dataframe)
    meanlap_df.insert(loc=len(meanlap_df.columns), column='Cluster',
                      value=clust)  # insert column back(end of dataframe)
    meanlap_df.insert(loc=len(meanlap_df.columns) - 1, column='label',
                      value=lbl)  # insert column back(end of dataframe)
    # laplace_df.columns = [*laplace_df.columns[:-1], 'Laplace'] #rename column with laplace noise
    return meanlap_df  # returns laplace dataframe


def modified_condensation_P2(data, group_size, epsilon):
    row_id = data.iloc[:, 0]  # pass row id column to hold
    data.drop(data.columns[0], axis=1, inplace=True)  # drop column from dataframe
    label = data.iloc[:, -1]  # pass label column to hold
    data.drop(data.columns[-1], axis=1, inplace=True)  # drop column from dataframe
    mini = min(data)  # get the minimum for each column
    mini_array = mini.to_numpy()  # minimum dataframe to array
    data_array = data.to_numpy()  # input dataframe to array
    dis = distance.cdist(data_array, mini_array, 'euclidean')  # calculating distance from minimum
    dis = np.append(dis, data_array, axis=1)  # append distance to the data
    dis_df = pd.DataFrame(dis)  # data df with distance
    dis_df.insert(loc=0, column='row_id', value=row_id)  # insert column back(front of dataframe)
    dis_df.insert(loc=len(dis_df.columns), column='label', value=label)  # insert column back(end of dataframe)
    dis = dis_df.to_numpy()  # convert back to array for sorting
    sorted_array = dis[dis[:, 1].argsort()]  # sort array based on second column(distance)
    sorted_df = pd.DataFrame(sorted_array)
    distance_df = clustering(sorted_df, group_size)  # split data in clusters
    distance_df.rename(columns={distance_df.columns[0]: "row_id"}, inplace=True)  # rename column 0 with row_id
    distance_df.rename(columns={distance_df.columns[1]: "distance"}, inplace=True)  # rename column 1 with distance
    distance_df.rename(columns={distance_df.columns[len(distance_df.columns) - 2]: "label"},
                       inplace=True)  # rename last column with label
    distance_df['row_id'] = distance_df['row_id'].astype(int)  # make row id column integers
    distance_df['Cluster'] = distance_df['Cluster'].astype(int)  # make cluster column integers
    mean_df = mean_calc(distance_df, group_size, epsilon)  # calculate mean, ptp for each cluster
    mean_df.drop(['distance', 'Cluster'], axis=1, inplace=True)  # drops columns distance, cluster
    return mean_df


##modified condensation P Code ends##

# accuracy checker code:
def compareacc(dfTrain, test_file, perturbed_file):
    # dfTrain=pd.read_csv(train_file,sep = ",",header=None)
    # dfTrain = readDataset(inputfile, headerExists)
    # if (len(dfTrain.columns) == 1):
    #     dfTrain = pd.read_csv(inputfile, sep="\t", header=None)
    cs = len(dfTrain.columns)
    print(cs)
    dfTrainData = dfTrain.iloc[:, :cs - 1]
    dfTrainData = dfTrainData.drop(dfTrainData.columns[[0, 1]], axis=1)
    # dfTrainData.iloc[:,0].replace(r"\.","",inplace = True,regex = True)
    # dfTrainData.iloc[:, 1].replace(r"\.", "", inplace=True,regex = True)
    dfTrainLabel = dfTrain.iloc[:, cs - 1]
    np.savetxt('train-label.txt', dfTrainLabel, delimiter='\n')
    dfTrain2 = pd.read_csv(perturbed_file, sep=",")
    dfTrainData2 = dfTrain2.iloc[:, 0:cs - 1]
    dfTrainData2 = dfTrainData2.drop(dfTrainData2.columns[[0, 1]], axis=1)
    # dfTrainData2.iloc[:, 0].replace(r"\.", "", inplace=True,regex = True)
    # dfTrainData2.iloc[:, 1].replace(r"\.", "", inplace=True,regex = True)

    dfTrainLabel2 = dfTrain2.iloc[:, cs - 1]
    np.savetxt('train-label-2.txt', dfTrainLabel2, delimiter='\n')
    # dfTrainLabel2.to_csv('label2.txt', sep=',',index=False)
    # dfTrainLabel.to_csv('label.txt', sep=',',index=False)

    dfTest = pd.read_csv(test_file, sep=",")
    if (len(dfTest.columns) == 1):
        dfTest = pd.read_csv(test_file, sep=",", header=None)
    dfTestData = dfTest.iloc[:, :cs - 1]
    dfTestData = dfTestData.drop(dfTestData.columns[[0, 1]], axis=1)
    # dfTestData.iloc[:, 0].replace(r"\.", "", inplace=True,regex = True)
    # dfTestData.iloc[:, 1].replace(r"\.", "", inplace=True,regex = True)
    dfTestLabel = dfTest.iloc[:, cs - 1]
    np.savetxt('test-label.txt', dfTestLabel, delimiter='\n')
    md1 = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    md1.fit(dfTrainData, dfTrainLabel)
    #c1 stores label using original data
    c1 = md1.predict(dfTestData)
    np.savetxt('predicted-label.txt', c1, delimiter='\n')
    md2 = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    # md2 = KNeighborsRegressor(n_neighbors=1, weights='distance')
    md2.fit(dfTrainData2, dfTrainLabel2)
    # c2 stores label using anonymized data
    c2 = md2.predict(dfTestData)
    np.savetxt('perturbed-label.txt', c2, delimiter='\n')
    #dfTestLabel is ground truth for test data
    v1 = c1 == dfTestLabel
    v2 = c2 == dfTestLabel
    lenV1 = 0
    lenV2 = 0
    for i in range(len(v1)):
        if v1[i] == True:
            lenV1 = lenV1 + 1
    for i in range(len(v2)):
        if v2[i] == True:
            lenV2 = lenV2 + 1
    accuracyorig = lenV1 / len(dfTestData)
    accuracycondense = lenV2 / len(dfTestData)
    print("accuracyorig = " + str(accuracyorig))
    print("accuracycondense = " + str(accuracycondense))

    FP1 = 0
    TP1 = 0
    FN1 = 0
    TN1 = 0
    FP2 = 0
    TP2 = 0;
    FN2 = 0;
    TN2 = 0;
    # 2 is attack, 1 is normal
    for i in range(len(dfTestLabel)):
        if (dfTestLabel[i] == 2 and c2[i] == 2):
            TP2 = TP2 + 1;
        if (dfTestLabel[i] == 2 and c2[i] == 1):
            FN2 = FN2 + 1;
        if (dfTestLabel[i] == 1 and c2[i] == 1):
            TN2 = TN2 + 1;
        if (dfTestLabel[i] == 1 and c2[i] == 2):
            FP2 = FP2 + 1;
        if (dfTestLabel[i] == 2 and c1[i] == 2):
            TP1 = TP1 + 1;
        if (dfTestLabel[i] == 2 and c1[i] == 1):
            FN1 = FN1 + 1;
        if (dfTestLabel[i] == 1 and c1[i] == 1):
            TN1 = TN1 + 1;
        if (dfTestLabel[i] == 1 and c1[i] == 2):
            FP1 = FP1 + 1;

    #hitRate = (TP + TN) / (TP + TN + FP + FN);
    TPRateOrig = TP1 /(FN1+TP1);
    FPRateOrig = FP1 / (TN1 + FP1);
    TPRateCond = TP2 / (FN2 + TP2);
    FPRateCond = FP2 / (TN2 + FP2);

    print("Original True positive rate = " + str(TPRateOrig))
    print("Original False positive rate = " + str(FPRateOrig))
    print("Condense True positive rate = " + str(TPRateCond))
    print("COndense False positive rate = " + str(FPRateCond))

"""In the following function, I have made some changes. We do not read the dataset here anymore. 
we read it in the input section, so that we can call either IPv4 or IPv6 functions from there. 
And I have made it into 2 functions, so that it becomes easier to call for both IPv4 and IPv6."""


def callIPv4Ops(df, numInstances):
    # Anonymize IP
    updateMapsIPv4(df)
    dfReplaced = replaceorIPv4(df)
    getInfoIPv4(dfReplaced)
    df = aggregationIPv4(dfReplaced, numInstances)
    # IP Anonymization Ends

    return df


def condense_perclass(df11, df_org, epsilon, t, wt, ite, priv, acc, labelcolId, anon_fields):
    # Anonymized source and destination IPs will be used for displaying results to the user with the '.'
    src_ip_anonymized = df11.iloc[:, 0]  # df1['SRC_IP']
    dest_ip_anonymized = df11.iloc[:, 1]  # df1['DST_IP']

    df1 = pd.DataFrame()
    # add a faked label column if no label in original data
    if (labelcolId == "n" or labelcolId == "N"):
        df11[len(df11.columns)] = 1
    # print('df11 ', len(df11.columns))
    # drop ip address, should be done after selecting columns because otherwise column index in original data will be incorrect
    df1 = df11.drop(df11.columns[[0, 1]], axis=1)  # df1.drop(columns=['SRC_IP', 'DST_IP'])
    cs = len(df1.columns)
    rs = len(df1)
    # print('df1', df1)
    print("new column length:" + str(cs))
    # class column should not be normalized
    # df1 = df1.drop(df1.columns[cs-1],axis = 1)

    normalizedData = pd.DataFrame(np.zeros((rs, cs - 1)))
    normalizer = []
    O_minVal = []
    O_maxVal = []
    for i in range(cs - 1):
        O_minVal.append(np.min(df1.values[:, i]))
        O_maxVal.append(np.max(df1.values[:, i]))
    rowidx = np.zeros(rs, dtype=np.int16)
    for i in range(rs):
        rowidx[i] = i
        # dest_ip[i] = dest_ip[i].replace(".","")
        # src_ip[i] = src_ip[i].replace(".","")
    df2 = pd.DataFrame(rowidx)
    df = pd.concat([df2, df1], axis=1, ignore_index=True)
    # print (df.iloc[:,cs].values.tolist())
    num_of_class = df.iloc[:, cs].max()

    print("num of class: " + str(num_of_class))
    csize = np.zeros(num_of_class, dtype=np.int16)
    for i in range(num_of_class):
        tmp1 = np.where(df.iloc[:, cs] == i + 1)  # class id starts from 1 not 0
        csize[i] = len(tmp1[0])
    print("csize: " + str(csize))
    tres_csize = np.zeros(num_of_class, dtype=np.int)
    for i in range(0, num_of_class):
        tres_csize[i] = np.floor(csize[i] / int(t))
        # print(tres_csize[i])
    print("tres_csize: " + str(tres_csize))
    size_of_subgrp = np.gcd(tres_csize[0], tres_csize[1])

    if num_of_class > 2:
        for i in range(1, num_of_class):
            size_of_subgrp = np.gcd(size_of_subgrp, tres_csize[i])

    size_of_subgrp = size_of_subgrp * t
    print("Cluster size: " + str(size_of_subgrp))
    group_size = np.floor(rs / size_of_subgrp)

    result = pd.DataFrame()

    for i in range(num_of_class):
        temp_data = pd.DataFrame()
        cellVal = np.where(df.iloc[:, cs] == i + 1)
        temp_data = temp_data.append(df.iloc[cellVal[0], :], ignore_index=True)
        temp_rs = len(temp_data)
        group_size = np.floor(temp_rs / size_of_subgrp)  # number_of_subgrp changed to group_size
        # print('group size = ',group_size)
        tempres = modified_condensation_P2(temp_data,size_of_subgrp, epsilon)  # number_of_subgrp = group_size
        result = result.append(tempres, ignore_index=True)
    print('tempres len ' + str(len(tempres)))
    sortedResult = result.sort_values('row_id')
    row_id = sortedResult['row_id']
    sortedResult.drop(sortedResult.columns[0], axis=1, inplace=True)

    # equation for normalization: (x-P_min)/(P_max-P_min)*(O_max-O_min) + O_min
    # normalized Data
    # classVal = rawdata.iloc[:, cs - 1]
    for i in range(0, cs - 1):
        P_minVal = float((np.min(sortedResult.values[:, i])))
        P_maxVal = float((np.max(sortedResult.values[:, i])))
        normalizer.append(P_maxVal - P_minVal)
        if normalizer[i] > 0:
            normalizedData.values[:, i] = ((sortedResult.values[:, i] - P_minVal) / (normalizer[i])) * (
                    O_maxVal[i] - O_minVal[i]) + O_minVal[i]
        if normalizer[i] == 0:
            normalizedData.values[:, i] = 0

    normalizedData.to_csv('perturbed-normalized.txt', sep=',', index=False)

    # sortedResult = pd.concat([sortedResult,classVal], axis=1, ignore_index=True)
    sortedResult = pd.concat([dest_ip_anonymized, sortedResult], axis=1, ignore_index=True)
    sortedResult = pd.concat([src_ip_anonymized, sortedResult], axis=1, ignore_index=True)
    sortedResult = pd.concat([row_id, sortedResult], axis=1, ignore_index=True)
    # sortedResult = sortedResult.sort_index(axis=0)
    sortedResult = sortedResult.sort_values(sortedResult.columns[0])
    df1 = pd.concat([dest_ip_anonymized, df1], axis=1, ignore_index=True)
    df1 = pd.concat([src_ip_anonymized, df1], axis=1, ignore_index=True)
    sortedResult.to_csv('perturbed-withrowid.txt', sep=',', index=False)

    sortedresultnorowid = sortedResult.drop(sortedResult.columns[0], axis=1)
    # print(sortedresultnorowid)
    sortedresultnorowid.to_csv('perturbed.txt', sep=',', index=False)
    sortedresultdataonly = sortedResult.iloc[:, 3:cs + 2];
    sortedresultdataonly.to_csv('perturbed-dataonly.txt', sep=',', index=False, header=False)
    rawdata = df1.iloc[:, 2:cs + 1]

    # print("df1\n", df1)

    rawdata.to_csv('rawdata.txt', sep=',', index=False, header=False)
    if (priv == "y"):
        privacy('rawdata.txt', 'perturbed-dataonly.txt', 'privacy.txt', 0.95)

    """write file remake code here"""
    cs1 = len(df_org.columns)

    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df_per = read_csv("perturbed.txt", sep=",")

    j = 0
    for i in range(0, cs1):
        # print(i)
        if i in anon_fields:
            df2 = pd.concat([df2, pd.DataFrame(df_per.iloc[:, j])], axis=1, ignore_index=True)
            df3 = pd.concat([df3, pd.DataFrame(df11.iloc[:, j])], axis=1, ignore_index=True)
            j += 1
        else:
            df2 = pd.concat([df2, pd.DataFrame(df_org.iloc[:, i])], axis=1, ignore_index=True)
            df3 = pd.concat([df3, pd.DataFrame(df_org.iloc[:, i])], axis=1, ignore_index=True)

    df2.to_csv('remade_perturbed.txt', sep=',', index=False)
    # print("df2\n", df2)
    ''''''

    # call it with df1

    if (acc == "y"):
        if IP_version == 4:
            compareacc(df3, 'unsw-test-small.csv', 'remade_perturbed.txt')
            #compareacc(df3, 'unsw-test.csv', 'remade_perturbed.txt')
        else:
            compareacc(df3, 'test-ipv6.csv', 'remade_perturbed.txt')


""" Accepts input from user."""


def getNonBlankInput(message, error_message):
    x = input(message)
    while len(x.strip()) == 0:
        x = input(error_message)

    return x


# generates dataframe and the list of fields that need to be anonymized based on contexts
def createDFtoWorkOn(df, colList, labelcolId, useContext):
    anon_fields = []
    cs = len(df.columns)
    if (labelcolId != "n" and labelcolId != "N"):
        colList = colList + " " + labelcolId

    num_list = list(int(num) for num in colList.strip().split(' '))

    if useContext == 'y' or useContext == 'Y':
        total_count = len(df)
        # print(total_count)

        for i in range(0, cs):
            if i in num_list:
                if (i == cs - 1 and labelcolId != "n" and labelcolId != "N"):
                    break

                column = df.iloc[:, i]
                unique_items = Counter(column)
                # print('i = ', i, ' ', len(unique_items))

                for key in unique_items:
                    value = unique_items[key]
                    percentage = value / total_count
	#				print(i)
    #               print(percentage)

                    if (percentage < cut_threshold):
                        anon_fields.append(i)
                        break

        # print(anon_fields)
        if (labelcolId != "n" and labelcolId != "N"):
            anon_fields.append(int(labelcolId))

        df2 = pd.DataFrame()
        for i in anon_fields:
            # print(i)
            df2 = pd.concat([df2, pd.DataFrame(df.iloc[:, i])], axis=1, ignore_index=True)

        return df2, anon_fields

    else:
        df1 = pd.DataFrame()
        for i in range(0, cs):
            if i in num_list:
                # print('i = ',i)
                df1 = pd.concat([df1, pd.DataFrame(df.iloc[:, i])], axis=1, ignore_index=True)
        return df1, num_list


def acceptUserInputs():
    #fileName = "unsw-train-100k.csv"  # getNonBlankInput("Please enter the path of the file", "file is mandatory")
    fileName = "unsw-train-1k.csv"
    #fileName = "sample_Ipv6_header.csv"
    #fileName = "data_2_full_test_weka.csv"
    eps =  1 # getNonBlankInput("Please enter the epsilon value", "epsilon value is mandatory")
    numInstances = 20  # getNonBlankInput("Please enter the number of instances per cluster", "number of instances per cluster value is mandatory")
    wt = 0  # getNonBlankInput("Please enter the weight", "weight value is mandatory")
    ite = 5  # getNonBlankInput("Please enter the number of iterations", "iterations value is mandatory")
    sep = ','  # getNonBlankInput("Please enter the separator (, or \\t) for the dataset", "separator value is mandatory")
    priv = 'y'  # getNonBlankInput("Do you wish to compute privacy (y/n)", "Please enter y if you wish to calculate privacy, else enter n")

    acc = 'n'  # getNonBlankInput("Do you wish to compute accuracy (y/n)", "Please enter y if you wish to calculate accuracy, else enter n")
    headerExists = 'y'  # getNonBlankInput("Does header exist (y/n)", "Please enter y if the data file has a header, else enter n")
    #colList = '0 1 2 3 4 5 6 7 8 9 10 11 12'
    #colList = '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81'
    #colList = '0 1 2 3'
    #colList = '0 1 3 8 9 10 14 16 39 44 45 46 50 56 57 58 67 69 70 71'
    colList = '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41'
    #colList = '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21'
    # getNonBlankInput("Please enter the column indexes, except the label column index separated by space(start with 0):", "Please enter the column indexes, except the label column index separated by space(start with 0):")
    labelcolId = '42'
    #labelcolId = '82'  # getNonBlankInput("Please enter the index of the label column, if there is no column for label, press n", "Please enter the index of the label column, if there is no column for label, press n")
    # possible extra input from users about context
    #useContext = 'y'  # getNonBlankInput("Do you want to do context based anonymization (y/n)", "Please enter y if yes, else enter n")
    useContext = 'n'
    # enter column id of the label column
    # if there is no class label then just anonymize and treat everything as one class
    #pr = profile.Profile()
    #pr.disable()
    #pr.enable()
    df = readDataset(fileName, sep, headerExists)
    print(df.iloc[0,:])
    # print(df)
    col1 = df.iloc[:, 0]
    ip1 = col1[0]
    df1 = None
    global IP_version
    if validateIPv4(ip1):
        print("inside if")
        df1 = callIPv4Ops(df, int(numInstances))
        IP_version = 4
        # print("inside if")
    elif validateIPv6(ip1):
        # print("inside else")
        IP_version = 6
        df1 = callIPv6Ops(df, int(numInstances))

    # print('df1 = ', len(df1.columns))
    print("createDFtoWorkOn")
    df2, anon_fields = createDFtoWorkOn(df1, colList, labelcolId, useContext)
    #df2.to_csv('data-short.txt', sep=',', index=False, header=False)
    # print('anon fields = ', anon_fields)
    # print('df2 = ', len(df2.columns))
    print("condense per class")
    condense_perclass(df2, df, float(eps), int(numInstances), float(wt), int(ite), priv, acc, labelcolId, anon_fields)
    #pr.disable()
    #ps = pstats.Stats(pr).strip_dirs().sort_stats('cumulative')
    #pr.print_stats(sort="cumulative")
    #pr.dump_stats('profile.pstat')
   # ps.print_callees()

acceptUserInputs()