import datetime
import pandas as pd


class dbook(object):
    def __init__(self):
        self.rating_data, self.user_feature, self.item_feature, self.user_user, self.rating_list = self.load()
        self.user_neighbor = self.get_neighbor()

    def load(self):
        input_dir = "data/dbook/original/"

        rating_data = pd.read_csv(input_dir + 'user_book.dat', names=['user', 'item', 'rating'], sep='\t', engine='python')

        rating_list = sorted(list(rating_data['rating'].unique()))
        print('rating types in user_book: %s' % len(rating_list))

        ul = pd.read_csv(input_dir + 'user_location.dat', names=['user', 'location'], sep='\t', engine='python')
        print('users in user_location: %s' % len(ul['user'].unique()))
        ug = pd.read_csv(input_dir + 'user_group.dat', names=['user', 'group'], sep='\t', engine='python')
        print('users in user_group: %s' % len(ug['user']))
        user_feature = ul
        print('user in user_feature: %s' % len(user_feature['user'].unique()))

        ba = pd.read_csv(input_dir + 'book_author.dat', names=['item', 'author'], sep='\t', engine='python')
        print('books in book_author: %s' % len(ba['item'].unique()))
        bp = pd.read_csv(input_dir + 'book_publisher.dat', names=['item', 'publisher'], sep='\t', engine='python')
        print('books in book_publisher: %s' % len(bp['item'].unique()))
        by = pd.read_csv(input_dir + 'book_year.dat', names=['item', 'year'], sep='\t', engine='python')
        print('books in book_year: %s' % len(by['item'].unique()))
        item_feature = pd.merge(ba, bp, on='item')
        print('books in book_feature: %s' % len(item_feature['item'].unique()))

        user_user = pd.read_csv(input_dir + 'user_user.dat', names=['user1', 'user2'], sep='\t', engine='python')

        return rating_data, user_feature, item_feature, user_user, rating_list

    def get_neighbor(self):
        user_neighbor = dict()
        for _, (row) in self.user_user.iterrows():
            user1 = row['user1']
            user2 = row['user2']
            if user1 not in user_neighbor:
                user_neighbor[user1] = []
            if user2 not in user_neighbor:
                user_neighbor[user2] = []
            user_neighbor[user1].append(user2)
            user_neighbor[user2].append(user1)
        return user_neighbor


class yelp(object):
    def __init__(self):
        self.rating_data, self.user_feature, self.item_feature, self.rating_list = self.load()
        self.user_neighbor = self.get_neighbor()

    def load(self):
        input_dir = "data/yelp/original/"

        rating_data = pd.read_csv(input_dir + 'rating.dat', names=['user', 'item', 'rating', 'time'], sep='\t', engine='python')
        rating_data = rating_data.drop(columns=['time'])

        rating_list = sorted(list(rating_data['rating'].unique()))
        print('rating types in yelp: %s' % len(rating_list))

        uf = pd.read_csv(input_dir + 'user_fans.dat', names=['user', 'fans'], sep='\t', engine='python')
        print('users in user_fan: %s' % len(uf['user'].unique()))
        ua = pd.read_csv(input_dir + 'user_avgrating.dat', names=['user', 'avgrating'], sep='\t', engine='python')
        print('users in user_avgrating: %s' % len(ua['user']))
        user_feature = pd.merge(uf, ua, on='user')
        print('user in user_feature: %s' % len(user_feature['user'].unique()))

        i_s = pd.read_csv(input_dir + 'item_stars.dat', names=['item', 'stars'], sep='\t', engine='python')
        print('items in item_stars: %s' % len(i_s['item'].unique()))
        i_p = pd.read_csv(input_dir + 'item_postalcode.dat', names=['item', 'postalcode'], sep='\t', engine='python')
        print('items in item_postalcode: %s' % len(i_p['item'].unique()))
        item_feature = pd.merge(i_s, i_p, on='item')
        print('item in item_feature: %s' % len(item_feature['item'].unique()))

        return rating_data, user_feature, item_feature, rating_list

    def get_neighbor(self):
        user_neighbor = dict()
        with open("data/yelp/original/user_friends.dat") as fin:
            for line in fin:
                data = line.strip().split('\t')
                if len(data) != 2:
                    continue
                user1 = int(data[0])
                user2s = list(map(int, data[1].split()))
                user_neighbor[user1] = user2s
        return user_neighbor


if __name__ == '__main__':
    # dbook()
    yelp()