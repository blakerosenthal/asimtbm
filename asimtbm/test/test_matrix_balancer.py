from asimtbm.utils.matrix_balancer import Balancer
import numpy as np
import pandas as pd


def test_balancer():
    """Verify that we get the same results as the test case
    in https://github.com/Dirguis/ipfn
    """
    m = np.array([1., 2., 1., 3., 5., 5., 6., 2., 2., 1., 7., 2.,
                  5., 4., 2., 5., 5., 5., 3., 8., 7., 2., 7., 6.], )
    dma_l = [501, 501, 501, 501, 501, 501, 501, 501, 501, 501, 501, 501,
             502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502]
    size_l = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
              1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]

    age_l = ['20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45']

    # create DataFrame to balance
    df = pd.DataFrame()
    df['dma'] = dma_l
    df['size'] = size_l
    df['age'] = age_l
    df['total'] = m

    # create aggregates
    dma = df.groupby('dma')['total'].sum()
    size = df.groupby('size')['total'].sum()
    age = df.groupby('age')['total'].sum()
    dma_size = df.groupby(['dma', 'size'])['total'].sum()
    size_age = df.groupby(['size', 'age'])['total'].sum()

    # alter aggregates
    dma.loc[501] = 52
    dma.loc[502] = 48

    size.loc[1] = 20
    size.loc[2] = 30
    size.loc[3] = 35
    size.loc[4] = 15

    age.loc['20-25'] = 35
    age.loc['30-35'] = 40
    age.loc['40-45'] = 25

    dma_size.loc[501] = [9, 17, 19, 7]
    dma_size.loc[502] = [11, 13, 16, 8]

    size_age.loc[1] = [7, 9, 4]
    size_age.loc[2] = [8, 12, 10]
    size_age.loc[3] = [15, 12, 8]
    size_age.loc[4] = [5, 7, 3]

    aggregates = [dma, size, age, dma_size, size_age]

    # dimensions to balance must match aggregates
    dimensions = [['dma'], ['size'], ['age'], ['dma', 'size'], ['size', 'age']]

    balancer = Balancer(df, aggregates, dimensions, weight_col='total',
                        convergence_rate=1e-6, max_iteration=50, closure=1e-4)
    balanced_df = balancer.balance()

    balanced_size = balanced_df.groupby('size')['total'].sum()
    both = pd.concat([balanced_size, size])
    diff = both.loc[balanced_size.index.symmetric_difference(size.index)]
    assert diff.empty
