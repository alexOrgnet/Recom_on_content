import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации из ALS"""

    def __init__(self, data, weight=True):
        """:data pd.DataFrame: Матрица взаимодействий user-item"""
        self.user_item_matrix = self.prepare_matrix(data)
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        if weight:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):
        return pd.pivot_table(data,
                              columns='item_id',
                              index='user_id',
                              aggfunc='count',
                              values='quantity',
                              fill_value=0.0).astype(float)

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Вспомогательные словари"""

        itemids = user_item_matrix.columns.values
        userids = user_item_matrix.index.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучение модели рекомендации товары, используя купленные"""

        recom = ItemItemRecommender(K=1)
        recom.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, factors=20, regularization=0.001, iterations=15):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=factors,
                                        regularization=regularization,
                                        iterations=iterations)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def get_recommendations(self, user, model, N=5):
        """Получить рекомендации товаров для пользователя
        :user: id пользователя
        :N: кол-во товаров в рекомендации"""

        res = [self.id_to_itemid[rec[0]] for rec in
                   model.recommend(userid=self.userid_to_id[user],
                                   user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                   N=N,
                                   filter_already_liked_items=False,
                                   filter_items=None,
                                   recalculate_user=True)]
        return res

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендации товаров, похожих на ранее купленные
        :user: id пользователя
        :N: кол-во товаров в рекомендации"""
        top = self.get_recommendations(user, self.own_recommender, N=N)

        res = []
        for item in top:
            similar_item = [self.id_to_itemid[rec[0]] for rec in
                            self.model.similar_items(self.itemid_to_id[item], N=2)[1:]]
            res.extend(similar_item)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендации товаров, похожих на ранее купленные другими пользователями
        :user: id пользователя
        :N: кол-во товаров в рекомендации"""
        similar_users = [self.id_to_userid[rec[0]] for rec in
                         self.model.similar_users(self.userid_to_id[user], N=N+1)[1:]]

        res = []
        for user in similar_users:
            # самый частопокупаемый товар каждого похожего пользователя
            item = self.get_recommendations(user, self.own_recommender, N=1)
            res.extend(item)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res