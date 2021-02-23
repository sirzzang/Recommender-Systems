import pandas as pd
pd.set_option('display.max_columns', None)

def topn_recommendations(user_num, item_df, user_item_df, user_pred_df, n=10):
    '''
    :param user_num: 추천을 제공할 user 번호
    :param item_df: 원래 item의 정보
    :param user_item_df: 원래 user-item 행렬
    :param user_pred_df: user의 rating에 대한 예측 행렬
    :param n: top N개 추천 시 N개
    :return: 상위 N개에 대한 추천 결과
    '''

    print(f'user {user_num}에 대한 추천 결과를 찾아 옵니다.')

    # top N item lists
    user_preds = user_pred_df.iloc[user_num-1, :].sort_values(ascending=False) # 내림차순으로 정렬
    user_preds.index += 1 # item index 1부터 시작하도록 조정

    # user history
    user_history = user_item_df.loc[user_num].dropna()

    # recommend movies that are not in user history
    user_recommendations = user_preds[~user_preds.isin(user_history)].sort_values(ascending=False)

    # top N recommendation results
    top_n_recommendations = pd.DataFrame(user_recommendations[:n]).reset_index()
    top_n_recommendations.columns = ['MovieID', 'Expected Ratings']

    # merge results with item info
    result = pd.merge(item_df, top_n_recommendations, on='MovieID') # TODO: column 이름 수작업

    return result