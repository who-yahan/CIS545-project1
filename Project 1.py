# project_1_ESE545
# Author: Xiangchen Dong, Yahan Liu
import csv
import numpy as np
import pandas as pd
import os
import random
import collections
import matplotlib.pyplot as plt
import sys
import time

# Problem 1

# Read txt file

file_path = os.path.expanduser('Netflix_data.txt')
x = pd.read_table(file_path, sep=',', engine='python', names=('user', 'rate', 'date'))
# Find index of the row containing ':', which is the MovieID
movie_row = x[x['user'].str.contains(":")]
# Check whether none are missing in MovieIDs and UserIDs
print("None are missing in MovieIDs and UserIDs is {}".format(
    len(x[pd.isna(x['date']) == True]) == len(x.iloc[list(movie_row.index)])))
# Only keep Rating 3 or above
no_movie_row = x.drop(list(movie_row.index))
no_movie_row.loc[no_movie_row['rate'] < 3, 'rate'] = 0
no_movie_row.loc[no_movie_row['rate'] >= 3, 'rate'] = 1
binary_no_movie_row = no_movie_row
# Remove users who rated more than or equal to 20
filtered = binary_no_movie_row.groupby("user").sum().apply(lambda n: n <= 20) & binary_no_movie_row.groupby(
    "user").sum().apply(lambda p: p >= 1)
# Find out which users are smaller than 0
selected_user_rows = list(filtered[filtered['rate'] == True].index)
rows_no_movie = binary_no_movie_row[binary_no_movie_row['user'].isin(selected_user_rows)]
# Eliminate rate = 0
final_rows_no_movie = rows_no_movie[rows_no_movie['rate'] != 0]
# Save into a dictionary, key is movie, value is list of users
data_dict = {}
for movie, [lowerbound, upperbound] in enumerate(
        zip(list(movie_row['user'].index), list(movie_row['user'].index)[1:] + [len(final_rows_no_movie) + 1])):
    value = final_rows_no_movie.loc[(final_rows_no_movie.index < upperbound) & (final_rows_no_movie.index > lowerbound)]
    user_list = list(value['user'])
    data_dict[movie] = user_list
print("convert into dict completed")
# Matrix saves as a form of np.array
sorted_username = sorted(list(final_rows_no_movie.groupby('user').count().index))
user_num = len(sorted_username)
matrix_shape = (len(movie_row), user_num)
print("matrix shape: ", matrix_shape)
matrix_output = np.zeros(shape=matrix_shape)
for movie in range(len(movie_row)):
    # Given a list of users, find the index in sorted_username
    users = data_dict[movie]
    index_list = [sorted_username.index(i) for i in sorted(users)]
    # Have the row to be 1 at index, else to be 0
    row_arr = matrix_output[movie]
    for ind in index_list:
        row_arr[ind] = 1
    if movie % 50 == 0:
        print("processing movie No. {}".format(movie))
export_path = os.path.join('Netflix_data.txt')
np.savetxt(os.path.expanduser(export_path), np.array(matrix_output), delimiter=',')
print(matrix_output)

# Problem 2

Jaccarddist = []
for i in range(0, 10000):
    # generate random pairs
    g = random.sample(range(0, user_num), 2)
    # print(g)
    # use them as the index and calculate the jaccard distance
    # print(list(matrix_output[:, g[0]]))
    # print(list(matrix_output[:, g[1]]))
    a = list(matrix_output[:, g[0]] + matrix_output[:, g[1]])
    Jaccardsim_new = a.count(1) / (a.count(1) + a.count(2))
    Jaccarddist_new = 1 - Jaccardsim_new
    Jaccarddist.append(Jaccarddist_new)
print(Jaccarddist)
print("average distance", sum(Jaccarddist) / len(Jaccarddist))
print("smallest distance", min(Jaccarddist))
# grouby and count of jaccard distance
Jaccarddist = sorted(Jaccarddist)
counter = collections.Counter(Jaccarddist)
Jaccarddist_hist = list(counter.keys())
Jaccarddist_freq = list(counter.values())
# print(Jaccarddist_hist)
# print(Jaccarddist_freq)
# draw histogram
plt.bar(Jaccarddist_hist, Jaccarddist_freq, width=0.1)
plt.xlabel('No. pairs')
plt.ylabel('Frequency')
plt.title('Histogram of random 10,000 pairs distance')
plt.show()
plt.savefig(os.path.expanduser('histogram.png'))

# Problem 3

eff_store = data_dict
for v in eff_store.values():
    if (row_arr[ind] != 1) in v:
        v.remove(v)

# Problem 4

def generate_params(n, p):
    params = []
    for i in range(n):
        a = np.random.randint(1, p)
        b = np.random.randint(1, p)
        params.append([a, b])
    return params


def make_minhash(char_mat):
    # row
    num_rows = len(char_mat)
    # cols
    num_cols = len(char_mat[0])
    print(num_cols)
    # generate funcs params
    funcs_params_list = generate_params(3, num_rows)
    print(funcs_params_list)
    # hash_func num
    func_length = len(funcs_params_list)
    sig_mat = np.zeros((func_length, num_cols))
    for k in range(func_length):
        for l in range(num_cols):
            sig_mat[k, l] = sys.maxsize
    for c in range(num_cols):
        for r in range(num_rows):
            if char_mat[r][c] == 1:
                for i, func_params in enumerate(funcs_params_list):
                    hash_func = lambda x: (func_params[0] * x + func_params[1]) % num_rows
                    sig_mat[i, c] = min(hash_func(r), sig_mat[i, c])
    return sig_mat


Sig_Matrix = make_minhash(matrix_output)
print(Sig_Matrix)
[Sig_row, Sig_column] = Sig_Matrix.shape
row_per_band = 1
band = 3


def Band2String(band_index, row_per_band, user_index):
    Band_Name = 0
    Band = Sig_Matrix[band_index * row_per_band:(band_index + 1) * row_per_band, user_index]
    for item in range(row_per_band):
        Band_Name += Band[item]
    Band_Name = str(band_index) + '.' + str(Band_Name)
    return (Band_Name)


def addBand(Dictionary, Band, User_ID):
    Dictionary.setdefault(Band, []).append(User_ID)


# Define a function to filter the dictionary
def filBand(dictionary):
    for key in list(dictionary.keys()):
        if len(dictionary[key]) < 2:
            del dictionary[key]


def addPair(imp_dict, exp_dict):
    for key in list(imp_dict.keys()):
        for item in imp_dict[key]:
            for other in imp_dict[key]:
                if other == item:
                    continue
                exp_dict.setdefault(item, []).append(other)


for band_index in range(band):
    New_Band_Dict = {}
    for user_index in range(Sig_column):
        addBand(New_Band_Dict, Band2String(band_index, row_per_band, user_index), user_index)
    Similar_pair = {}
    filBand(New_Band_Dict)
    addPair(New_Band_Dict, Similar_pair)
    print("recording process:", 100 * band_index / band, "%")
print("finish recording the similar pairs")
with open('similarPairs.csv', 'w') as writeFile:
    similarWriter = csv.writer(writeFile, delimiter=',')
    for i in range(len(Similar_pair)):
        listPairs = list(Similar_pair[i])
        for j in range(len(listPairs)):
            similarWriter.writerow([i, listPairs[j]])


# Problem 5

def get_sig(num, data, r):
    sig = np.empty((num, data.shape[1]), dtype=int)
    for i in range(0, num):
        a = random.randint(0, r - 1)
        b = random.randint(0, r - 1)
        for col in range(0, data.shape[1]):
            tempMin = data.shape[0]
            for row in range(0, data.shape[0]):
                if data[row][col] == 1:
                    tempHash = (a * row + b) % r
                    tempMin = min(tempMin, tempHash)
            sig[i][col] = tempMin
    return sig


def get_sig_dic(hash_num, user_dic, prime):
    sig = np.empty((hash_num, len(user_dic)), dtype=int)

    for i in range(0, hash_num):
        a = random.randint(0, prime - 1)
        b = random.randint(0, prime - 1)
        for user in user_dic.keys():
            sig[i][user] = min(list(map(lambda x: (x * a + b) % prime, user_dic[user])))
    return sig


def find_sim_dic(sig, thre, r, prime, sorted_username, self=None):
    buckets = []
    bands = int(sig.shape[0] / r)
    print(str(bands) + " bands to be processed.")
    for i in range(0, bands):
        bucket = {}
        a = [random.randint(0, prime) for i in range(r)]
        b = [random.randint(0, prime) for i in range(r)]
        for j in range(0, sig.shape[1]):
            s = time.time()
            tempHash = 0;
            for k in range(r):
                tempHash += (sig[i * r:(i + 1) * r, j] * a[k] + b[k]) % prime
            # print("time line 1: "+ str(time.time()-s))
            tempHash = tuple(tempHash)
            s = time.time()
            bucket.setdefault(tempHash, []).append(j)
        # print("time line 2: "+ str(time.time()-s))
        buckets.append(bucket)
        print("band {} completed.".format(str(i + 1)))
    start = time.time()
    candidates = set()
    a = 1
    for bucket in buckets:
        print("Looking for candidate pairs in bucket ", str(a))
        for l in bucket.values():
            size = len(l)
            # print("Current Group size" + str(len(l)))
            if size == 1:
                continue;
            # set up comparison pairs for each two elements
            for i in range(size):
                for j in range(i + 1, size):
                    candidates.add((l[i], l[j]))
                    # candidates.add((l[i],l[j])) where l[i] is the index of users
        a += 1
    print(str(len(candidates)) + " condidates found!")
    print("time to go through for loops to find pairs: {}".format(time.time() - start))
    start = time.time()
    final_pairs = []
    final_pairs_ind = []
    for (i, j) in candidates:
        count = sum(sig[:, i].reshape(-1) == sig[:, j].reshape(-1))
        if float(count) / float(sig.shape[0]) >= thre:
            final_pairs.append((sorted_username[i], sorted_username[j]))
            final_pairs_ind.append((i, j))
    print("final pairs takes {} seconds".format(time.time() - start))