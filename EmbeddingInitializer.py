import torch
from torch.autograd import Variable
# import torchsnooper


# Yelp dataset
class UserEmbeddingYelp(torch.nn.Module):
    def __init__(self, config):
        super(UserEmbeddingYelp, self).__init__()
        self.num_fans = config['num_fans']
        self.num_avgrating = config['num_avgrating']
        self.embedding_dim = config['embedding_dim']

        self.embedding_fans = torch.nn.Embedding(
            num_embeddings=self.num_fans,
            embedding_dim=self.embedding_dim
        )

        self.embedding_avgrating = torch.nn.Embedding(
            num_embeddings=self.num_avgrating,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        fans_idx = Variable(user_fea[:, 0], requires_grad=False)  # [#sample]
        avgrating_idx = Variable(user_fea[:, 1], requires_grad=False)  # [#sample]
        fans_emb = self.embedding_fans(fans_idx)
        avgrating_emb = self.embedding_avgrating(avgrating_idx)
        return torch.cat((fans_emb, avgrating_emb), 1)   # (1, 1*32)


class ItemEmbeddingYelp(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingYelp, self).__init__()
        self.num_stars = config['num_stars']
        self.num_postalcode = config['num_postalcode']
        self.embedding_dim = config['embedding_dim']

        self.embedding_stars = torch.nn.Embedding(
            num_embeddings=self.num_stars,
            embedding_dim=self.embedding_dim,
        )

        self.embedding_postalcode = torch.nn.Embedding(
            num_embeddings=self.num_postalcode,
            embedding_dim=self.embedding_dim,
        )

    def forward(self, item_fea):
        stars_idx = Variable(item_fea[:, 0], requires_grad=False)
        postalcode_idx = Variable(item_fea[:, 1], requires_grad=False)

        stars_emb = self.embedding_stars(stars_idx)  # (1,32)
        postalcode_emb = self.embedding_postalcode(postalcode_idx)  # (1,32)
        return torch.cat((stars_emb, postalcode_emb), 1)


# DBook dataset
class UserEmbeddingDB(torch.nn.Module):
    def __init__(self, config):
        super(UserEmbeddingDB, self).__init__()
        self.num_location = config['num_location']
        self.embedding_dim = config['embedding_dim']

        self.embedding_location = torch.nn.Embedding(
            num_embeddings=self.num_location,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """

        :param user_fea: tensor, shape = [#sample, #user_fea]
        :return:
        """
        location_idx = Variable(user_fea[:, 0], requires_grad=False)  # [#sample]
        location_emb = self.embedding_location(location_idx)
        return location_emb


class ItemEmbeddingDB(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingDB, self).__init__()
        self.num_publisher = config['num_publisher']
        self.embedding_dim = config['embedding_dim']

        self.embedding_publisher = torch.nn.Embedding(
            num_embeddings=self.num_publisher,
            embedding_dim=self.embedding_dim
        )

    def forward(self, item_fea):
        """
        :param item_fea:
        :return:
        """
        publisher_idx = Variable(item_fea[:, 1], requires_grad=False)
        publisher_emb = self.embedding_publisher(publisher_idx)  # (1,32)
        return publisher_emb  # (1, 1*32)


