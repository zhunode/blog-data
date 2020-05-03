#!/usr/bin/env python
# coding: utf-8

# # 深度之眼 - Kaggle 比赛训练营

# - **BaseLine步骤**：    
#                         1. 数据分析 EDA
#                         2. 特征工程
#                         3. 模型训练
#                         4. 线下验证

# ## 一、数据分析
# - 查看 sales 数据前几行
# - 查看 sales 数据聚合结果趋势
# - 查看 sales 数据标签分布
'''
首先，每一条商品为一个时间序列(Sale)，
其他维度信息：商品种类，店铺ID，所在州，价格，日期，节假日信息等。

A榜  1913天数据预测后28天数据： 2016-04-25 to 2016-05-22
B榜  1941天数据预测后28天数据： 2016-05-23 to 2016-06-19

按照不同维度对时间序列进行聚合
但是可以将30490条数据按照不同维度进行聚合：
1 我们可以将30490天数据聚合成一条数据，得到1条时间序列，即level 1。
2 可以按照州的维度进行聚合，因为有3个州，所以可以聚合得到3条数据，即得到3条时间序列，即level 2.
3 可以按照商店(store)的维度进行聚合，有10家商店，所以可以聚合得到10条数据，即level 3.
4 可以按照商品类别(category)的维度进行聚合，因为有3个类目，所以可以聚合得到3条数据，即level 4。
5 可以按照产品部门(department)的维度进行聚合，因为有7个部门，所以可以局和得到7条数据，得到level 5。
6 可以按照state和category维度组合聚合，得到3*3条数据，即level 6.
7 可以按照state和department维度组合聚合，得到3*7条数据，即level7。
8 可以按照store和category维度组合聚合，得到3*10条数据，得到level8。
9 可以按照store和department组合聚合，得到10*7条数据，即level 9。
10
11
12
为什们说时间序列聚合呢？因为我们最终的目的不是预测30490条数据，而是预测42840条数据的准确度。

接下来再看一下评估指标：
WRMSSE : weighted root mean squared scaled error
相当于是MSE前面加了一个权重。WRMSSE = \sum_{i=1}^{42840}{w_i \cdot RMSSE}
其中\sum_{i=1}^{42840}{w_i} = 1。归一化

问题：评估指标和损失函数的区别：
我们再机器学习训练模型的时候，希望损失值越小越好；而我们再来打比赛的时候，也希望
模型的评估指标的损失越小越好。他们的区别是评估指标不一定能够优化。从而通过损失函数替代评估指标，
如果能够直接用评估指标进行优化/求导，那么我们就用评估指标作为我们的损失函数，而若是不能优化，我们
就需要找一个损失函数来替代我们的评估指标，对于这个评估指标而言，损失函数可能会替代的好，也可能表现差。
若损失函数能够替代评估指标，那么就表明在损失函数上表现好的模型在评估指标上也会得不错。


M5 eda得到的信息：
1 大量数据为0，且为整数
2 总体趋势向上，，具有明显的年周期性
3 按不同维度聚合，呈现不同趋势
'''

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', 100)

sale_data = pd.read_csv('./m5-forecasting-accuracy/sales_train_validation.csv')

sale_data.head(5)

day_data = sale_data[[f'd_{day}' for day in range(1, 1914)]]

# 统计商品店每一天的总销售情况
total_sum = np.sum(day_data, axis=0).values

# 这个统计的是商品的在时间窗口内的销售量
# total_item_sum = np.sum(day_data,axis=1).values

plt.plot(total_sum)

day_data[day_data < 100].values.reshape(-1)

# ## 二、特征工程
# 
# 选定机器学习的建模方案，核心思想是对时间序列抽取窗口特征。

# 抽取窗口特征：
# - 前7天
# - 前28天
# - 前7天均值
# - 前28天均值
# 
# 关联其他维度信息
# 
# - 日期
# - 价格


import sys
import lightgbm as lgb
from datetime import datetime, timedelta

'''
train_start : 指的是取多长的数据参与训练，没有必要让全部的数据参与训练。
test_start : 去多长的数据参与测试
is_train : 是训练集呢还是测试集

这里为什们不需要全部的样本参与训练呢？我个人觉得预测的销量应该是与其相近的销量相关的，历史越久远的销售量应该是越不相关的，甚至是干扰。 
'''
def create_train_data(train_start=750, test_start=1800, is_train=True):
    # 基本参数
    PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16", "sell_price": "float32"}
    CAL_DTYPES = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
                  "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
                  "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32'}

    start_day = train_start if is_train else test_start
    numcols = [f"d_{day}" for day in range(start_day, 1914)]
    catcols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
    SALE_DTYPES = {numcol: "float32" for numcol in numcols}

    # dict update 方法应该是拥有修改和添加的方法
    SALE_DTYPES.update({col: "category" for col in catcols if col != "id"})

    # 加载price数据
    price_data = pd.read_csv('./m5-forecasting-accuracy/sell_prices.csv', dtype=PRICE_DTYPES)

    # 加载cal数据
    cal_data = pd.read_csv('./m5-forecasting-accuracy/calendar.csv', dtype=CAL_DTYPES)

    # 加载sale数据
    sale_data = pd.read_csv('./m5-forecasting-accuracy/sales_train_validation.csv', dtype=SALE_DTYPES,
                            usecols=catcols + numcols)

    # 类别标签转换
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            print(price_data[col].cat)
            print(price_data[col].cat.codes)

            price_data[col] = price_data[col].cat.codes.astype("int16")
            price_data[col] -= price_data[col].min()
            break

    cal_data["date"] = pd.to_datetime(cal_data["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            # 对category类型特征进行编码
            cal_data[col] = cal_data[col].cat.codes.astype("int16")
            cal_data[col] -= cal_data[col].min()

    for col in catcols:
        if col != "id":
            sale_data[col] = sale_data[col].cat.codes.astype("int16")
            sale_data[col] -= sale_data[col].min()

    # 注意提交格式里有一部分为空
    if not is_train:
        for day in range(1913 + 1, 1913 + 2 * 28 + 1):
            sale_data[f"d_{day}"] = np.nan

    sale_data = pd.melt(sale_data,
                        id_vars=catcols,
                        value_vars=[col for col in sale_data.columns if col.startswith("d_")],
                        var_name="d",
                        value_name="sales")
    # 将销量数据、价格数据和calendar数据进行关联
    sale_data = sale_data.merge(cal_data, on="d", copy=False)
    sale_data = sale_data.merge(price_data, on=["store_id", "item_id", "wm_yr_wk"], copy=False)
    return sale_data


def create_feature(sale_data, is_train=True, day=None):
    # 可以在这里加入更多的特征抽取方法
    # 获取7天前的数据，28天前的数据 即将7/28作为时间窗口进行特征抽取
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]

    # 如果是测试集只需要计算一天的特征，减少计算量
    # 注意训练集和测试集特征生成要一致
    if is_train:
        for lag, lag_col in zip(lags, lag_cols):
            sale_data[lag_col] = sale_data[["id", "sales"]].groupby("id")["sales"].shift(lag)
    else:
        for lag, lag_col in zip(lags, lag_cols):
            sale_data.loc[sale_data.date == day, lag_col] = sale_data.loc[
                sale_data.date == day - timedelta(days=lag), 'sales'].values

            # 将获取7天前的数据，28天前的数据做移动平均
    wins = [7, 28]

    if is_train:
        for win in wins:
            for lag, lag_col in zip(lags, lag_cols):
                sale_data[f"rmean_{lag}_{win}"] = sale_data[["id", lag_col]].groupby("id")[lag_col].transform(
                    lambda x: x.rolling(win).mean())
    else:
        for win in wins:
            for lag in lags:
                df_window = sale_data[
                    (sale_data.date <= day - timedelta(days=lag)) & (sale_data.date > day - timedelta(days=lag + win))]
                df_window_grouped = df_window.groupby("id").agg({'sales': 'mean'}).reindex(
                    sale_data.loc[sale_data.date == day, 'id'])
                sale_data.loc[sale_data.date == day, f"rmean_{lag}_{win}"] = df_window_grouped.sales.values

                # 处理时间特征
    # 有的时间特征没有，通过datetime的方法自动生成
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in sale_data.columns:
            sale_data[date_feat_name] = sale_data[date_feat_name].astype("int16")
        else:
            sale_data[date_feat_name] = getattr(sale_data["date"].dt, date_feat_func).astype("int16")
    return sale_data


sale_data = create_train_data(train_start=350, is_train=True)
sys.exit()
sale_data = create_feature(sale_data)

# 清洗数据，选择需要训练的数据
sale_data.dropna(inplace=True)
cat_feats = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1",
                                                                        "event_type_2"]
useless_cols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]
train_cols = sale_data.columns[~sale_data.columns.isin(useless_cols)]
X_train = sale_data[train_cols]
y_train = sale_data["sales"]

print(X_train.head(5))

print(y_train.head(5))


# ## 三、模型训练
# 
# 选择 LGB 模型进行模型的训练。
'''

# - 损失函数的选择
# - 预测时候的技巧

这里的损失函数为什们选择tweedie损失呢？而不是MSE损失呢？
主要是因为标签的的分布符合泊松分布/tweedie分布，从而不能选择MSE损失，因为MSE是高斯分布。

# tweedie_variance_power 参数的选择 [1,2] 之间。
# tweedie分布中有一个参数p，当 p=1的时候，tweedie分布r就是泊松分布，当p=2时，tweedie分布为Gama分布。
# 在1和2之间，就靠近谁，就更像谁。


# LGB 模型是 GBDT 模型的变种，无法突然训练集的上界。
因为LGB是对空间的划分，无法突破原有空间内的上限值，而我们预测的销售额都是呈现上升趋势，

改进方案1：
那么就需要让LGB打破这个趋势，就通过在LGB前添加参数，如1.1倍的参数来进行拟合；或者是更大的参数进行拟合；这样就能够得到上升到趋势，
改进方案2：
将LGB的预测结果运用到时间序列模型，如趋势拟合profit模型等。

'''


def train_model(train_data, valid_data):
    params = {
        "objective": "tweedie", # 这里损失函数选择的是tweedie损失，而不是选择MSE损失。
        "metric": "rmse",
        "force_row_wise": True,
        "learning_rate": 0.075,
        "sub_feature": 0.8,
        "sub_row": 0.75,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        "metric": ["rmse"],
        "nthread": 8,
        "tweedie_variance_power": 1.1, # 这里将p值设置为1.1，当然也可以尝试其他值试试。
        'verbosity': 1,
        'num_iterations': 1500,
        'num_leaves': 128,
        "min_data_in_leaf": 104,
    }

    m_lgb = lgb.train(params, train_data, valid_sets=[valid_data], verbose_eval=50)

    return m_lgb


def predict_ensemble(train_cols, m_lgb):
    date = datetime(2016, 4, 25)
    # alphas = [1.035, 1.03, 1.025, 1.02]
    # alphas = [1.028, 1.023, 1.018]
    alphas = [1.035, 1.03, 1.025] # 这里用改进方案1进行预测，
    weights = [1 / len(alphas)] * len(alphas)
    sub = 0.

    test_data = create_train_data(is_train=False)

    '''
    预测过程：用第一天的预测结果，经过特征工程加工后，在预测第二天的数值；对二天的结果进行特征工程加工，预测得到第三天的数值。
    循环迭代往前预测的过程。
    '''
    for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

        test_data_c = test_data.copy()
        cols = [f"F{i}" for i in range(1, 29)]

        for i in range(0, 28):
            day = date + timedelta(days=i)
            print(i, day)
            tst = test_data_c[(test_data_c.date >= day - timedelta(days=57)) & (test_data_c.date <= day)].copy()
            tst = create_feature(tst, is_train=False, day=day)
            tst = tst.loc[tst.date == day, train_cols]

            test_data_c.loc[test_data_c.date == day, "sales"] = alpha * m_lgb.predict(tst)

        # 改为提交数据的格式
        test_sub = test_data_c.loc[test_data_c.date >= date, ["id", "sales"]].copy()
        test_sub["F"] = [f"F{rank}" for rank in test_sub.groupby("id")["id"].cumcount() + 1]
        test_sub = test_sub.set_index(["id", "F"]).unstack()["sales"][cols].reset_index()
        test_sub.fillna(0., inplace=True)
        test_sub.sort_values("id", inplace=True)
        test_sub.reset_index(drop=True, inplace=True)
        test_sub.to_csv(f"submission_{icount}.csv", index=False)
        if icount == 0:
            sub = test_sub
            sub[cols] *= weight
        else:
            sub[cols] += test_sub[cols] * weight
        print(icount, alpha, weight)

    sub2 = sub.copy()
    # 把大于28天后的validation替换成evaluation
    sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
    sub = pd.concat([sub, sub2], axis=0, sort=False)
    sub.to_csv("submissionV3.csv", index=False)


train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feats, free_raw_data=False)
valid_inds = np.random.choice(len(X_train), 10000)
valid_data = lgb.Dataset(X_train.iloc[valid_inds], label=y_train.iloc[valid_inds], categorical_feature=cat_feats,
                         free_raw_data=False)

m_lgb = train_model(train_data, valid_data)
predict_ensemble(train_cols, m_lgb)

# ## 四、线下验证
# 
# WRMSSE 的评估方法和 RMSE 很不一致，我们需要拆分出么一条时间序列的权重到底是多少，一方面能帮助我们做线下验证，另一方面可以帮助我们思考能否使用自定义的损失函数。


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
import gc


# 转换数据类型，减少内存占用空间
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


# 加载数据
data_pass = './m5-forecasting-accuracy/'

# sale数据
sales = pd.read_csv(data_pass + 'sales_train_validation.csv')

# 日期数据
calendar = pd.read_csv(data_pass + 'calendar.csv')
calendar = reduce_mem_usage(calendar)

# 价格数据
sell_prices = pd.read_csv(data_pass + 'sell_prices.csv')
sell_prices = reduce_mem_usage(sell_prices)

# 计算价格
# 按照定义，只需要计算最近的 28 天售卖量（售卖数*价格），通过这个可以得到 weight
# 可以不是 1914
cols = ["d_{}".format(i) for i in range(1914 - 28, 1914)]
data = sales[["id", 'store_id', 'item_id'] + cols]

# 从横表改为纵表
data = data.melt(id_vars=["id", 'store_id', 'item_id'], var_name="d", value_name="sale")

# 和日期数据做关联
data = pd.merge(data, calendar, how='left', left_on=['d'], right_on=['d'])

data = data[["id", 'store_id', 'item_id', "sale", "d", "wm_yr_wk"]]

# 和价格数据关联
data = data.merge(sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
data.drop(columns=['wm_yr_wk'], inplace=True)

# 计算售卖量
data['sale_usd'] = data['sale'] * data['sell_price']

# 得到聚合矩阵
# 30490 -> 42840
# 需要聚合的维度明细计算出来
dummies_list = [sales.state_id, sales.store_id,
                sales.cat_id, sales.dept_id,
                sales.state_id + sales.cat_id, sales.state_id + sales.dept_id,
                sales.store_id + sales.cat_id, sales.store_id + sales.dept_id,
                sales.item_id, sales.state_id + sales.item_id, sales.id]

# 全部聚合为一个， 最高 level
dummies_df_list = [pd.DataFrame(np.ones(sales.shape[0]).astype(np.int8),
                                index=sales.index, columns=['all']).T]

# 挨个计算其他 level 等级聚合
for i, cats in enumerate(dummies_list):
    dummies_df_list += [pd.get_dummies(cats, drop_first=False, dtype=np.int8).T]

# 得到聚合矩阵
roll_mat_df = pd.concat(dummies_df_list, keys=list(range(12)),
                        names=['level', 'id'])  # .astype(np.int8, copy=False)

# 保存聚合矩阵
roll_index = roll_mat_df.index
roll_mat_csr = csr_matrix(roll_mat_df.values)
roll_mat_df.to_pickle('roll_mat_df.pkl')

# 释放内存
del dummies_df_list, roll_mat_df
gc.collect()


# 按照定义，计算每条时间序列 RMSSE 的权重:
def get_s(drop_days=0):
    """
    drop_days: int, equals 0 by default, so S is calculated on all data.
               If equals 28, last 28 days won't be used in calculating S.
    """

    # 要计算的时间序列长度
    d_name = ['d_' + str(i + 1) for i in range(1913 - drop_days)]
    # 得到聚合结果
    sales_train_val = roll_mat_csr * sales[d_name].values

    # 按照定义，前面连续为 0 的不参与计算
    start_no = np.argmax(sales_train_val > 0, axis=1)

    # 这些连续为 0 的设置为 nan
    flag = np.dot(np.diag(1 / (start_no + 1)), np.tile(np.arange(1, 1914 - drop_days), (roll_mat_csr.shape[0], 1))) < 1
    sales_train_val = np.where(flag, np.nan, sales_train_val)

    # 根据公式计算每条时间序列 rmsse的权重
    weight1 = np.nansum(np.diff(sales_train_val, axis=1) ** 2, axis=1) / (1913 - start_no - 1)

    return weight1


S = get_s(drop_days=0)


# 根据定义计算 WRMSSE 的权重，这里指 w
def get_w(sale_usd):
    """
    """
    # 得到最细维度的每条时间序列的权重
    total_sales_usd = sale_usd.groupby(
        ['id'], sort=False)['sale_usd'].apply(np.sum).values

    # 通过聚合矩阵得到不同聚合下的权重
    weight2 = roll_mat_csr * total_sales_usd

    return 12 * weight2 / np.sum(weight2)


W = get_w(data[['id', 'sale_usd']])

SW = W / np.sqrt(S)

sw_df = pd.DataFrame(np.stack((S, W, SW), axis=-1), index=roll_index, columns=['s', 'w', 'sw'])
sw_df.to_pickle('sw_df.pkl')


# 评分函数
# 得到聚合的结果
def rollup(v):
    return (v.T * roll_mat_csr.T).T


# 计算 WRMSSE 评估指标
def wrmsse(preds, y_true, score_only=False, s=S, w=W, sw=SW):
    '''
    preds - Predictions: pd.DataFrame of size (30490 rows, N day columns)
    y_true - True values: pd.DataFrame of size (30490 rows, N day columns)
    sequence_length - np.array of size (42840,)
    sales_weight - sales weights based on last 28 days: np.array (42840,)
    '''

    if score_only:
        return np.sum(
            np.sqrt(
                np.mean(
                    np.square(rollup(preds.values - y_true.values))
                    , axis=1)) * sw * 12)
    else:
        score_matrix = (np.square(rollup(preds.values - y_true.values)) * np.square(w)[:, None]) * 12 / s[:, None]
        score = np.sum(np.sqrt(np.mean(score_matrix, axis=1)))
        return score, score_matrix


# 加载前面预先计算好的各个权重
file_pass = './'
sw_df = pd.read_pickle(file_pass + 'sw_df.pkl')
S = sw_df.s.values
W = sw_df.w.values
SW = sw_df.sw.values

roll_mat_df = pd.read_pickle(file_pass + 'roll_mat_df.pkl')
roll_index = roll_mat_df.index
roll_mat_csr = csr_matrix(roll_mat_df.values)

print(sw_df.loc[(11, slice(None))].sw)

np.max(sw_df.loc[(11, slice(None))].sw)
