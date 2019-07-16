import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionedCNNClassifier(nn.Module):
    def __init__(self, net_cfg, embed_cfg):
        super().__init__()
        self.net_cfg = net_cfg
        self.embed_cfg = embed_cfg
        print('----------- Model Config---------------')
        print(f'Headline Embedding Size: {self.embed_cfg["H_V"]}')
        print(f'Body Embedding Size: {self.embed_cfg["B_V"]}')
        print(f'Number of Classes: {self.net_cfg["num_classes"]}')
        print('---------------------------------------')
        self.h_embedding = nn.Embedding(self.embed_cfg['H_V'], self.embed_cfg['D'])
        self.b_embedding = nn.Embedding(self.embed_cfg['B_V'], self.embed_cfg['D'])
        self.convs_headline = nn.ModuleList(
                [self.n_gram_conv(n, self.net_cfg['h_num_filt'])
                    for n in self.net_cfg['h_n_list']])
        self.convs_body = nn.ModuleList(
                [self.n_gram_conv(n, self.net_cfg['b_num_filt'])
                    for n in self.net_cfg['b_n_list']])
        self.fc_out = nn.Sequential(
                nn.Linear(
                (len(self.net_cfg['b_n_list']) * self.net_cfg['b_num_filt']) +
                (len(self.net_cfg['h_n_list']) * self.net_cfg['h_num_filt']), 1024),
                nn.ReLU(),
                nn.Dropout(self.net_cfg['dropout_rate']),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(self.net_cfg['dropout_rate']),
                nn.Linear(256, self.net_cfg['num_classes'])
                )

    def n_gram_conv(self, n, num_filt):
        return nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = num_filt,
                kernel_size = (n, self.embed_cfg['D'])),
            nn.ReLU())

    def forward(self, h, b):
        # h = (Batch, Sentence Words)
        # b = (Batch, Sentence Words)
        h = self.h_embedding(h) # (Batch, Word, Vector)
        b = self.b_embedding(b) # (Batch, Word, Vector)
        h = h.unsqueeze(1) # (Batch, 1, Word, Vector)
        b = b.unsqueeze(1) # (Batch, 1, Word, Vector)

        # (Batch, Num_Filters, Num_Feature_Map, 1) * len(h_n_list)
        h_convs_out = [conv(h) for conv in self.convs_headline]

        # (Batch, Num_Filters, Num_Feature_Map, 1) * len(b_n_list)
        b_convs_out = [conv(b) for conv in self.convs_body]

        # (Batch, Num_Filters, Num_Feature_Map) * len(h_n_list)
        h_convs_out = [output.squeeze(3) for output in h_convs_out]

        # (Batch, Num_Filters, Num_Feature_Map) * len(b_n_list)
        b_convs_out = [output.squeeze(3) for output in b_convs_out]

        # (Batch, Num_Filters, 1) * len(h_n_list)
        # MaxPool1D: 2nd arg is kernel size
        # the stride is taken to be equal to kernel size by default
        h_convs_out = [F.max_pool1d(h_conv_out, h_conv_out.shape[2])
                       for h_conv_out in h_convs_out]

        # (Batch, Num_Filters, 1) * len(b_n_list)
        b_convs_out = [F.max_pool1d(b_conv_out, b_conv_out.shape[2])
                       for b_conv_out in b_convs_out]

        # (Batch, Num_Filters) * len(h_n_list)
        h_convs_out = [h_conv_out.squeeze(2) for h_conv_out in h_convs_out]

        # (Batch, Num_Filters) * len(h_n_list)
        b_convs_out = [b_conv_out.squeeze(2) for b_conv_out in b_convs_out]

        # (Batch, Num_Filters * len(h_n_list))
        h_feature_vec = torch.cat(h_convs_out, dim = 1)
        b_feature_vec = torch.cat(b_convs_out, dim = 1)

        h_b_ft = torch.cat([h_feature_vec, b_feature_vec], dim = 1)
        logits = self.fc_out(h_b_ft)
        return logits, h_feature_vec, b_feature_vec

class ConditionedSharedCNNClassifier(nn.Module):
    def __init__(self, net_cfg, embed_cfg):
        super().__init__()
        self.net_cfg = net_cfg
        self.embed_cfg = embed_cfg
        print('----------- Model Config---------------')
        print(f'Headline Embedding Size: {self.embed_cfg["H_V"]}')
        print(f'Body Embedding Size: {self.embed_cfg["B_V"]}')
        print(f'Number of Classes: {self.net_cfg["num_classes"]}')
        print('---------------------------------------')
        self.h_embedding = nn.Embedding(self.embed_cfg['H_V'], self.embed_cfg['D'])
        self.b_embedding = nn.Embedding(self.embed_cfg['B_V'], self.embed_cfg['D'])
        self.shared_convs = nn.ModuleList(
                [self.n_gram_conv(n, self.net_cfg['num_filt'])
                    for n in self.net_cfg['n_list']])
        self.fc_out = nn.Sequential(
                nn.Linear(
                (2 * len(self.net_cfg['n_list']) * self.net_cfg['num_filt']), 1024),
                nn.ReLU(),
                nn.Dropout(self.net_cfg['dropout_rate']),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(self.net_cfg['dropout_rate']),
                nn.Linear(256, self.net_cfg['num_classes'])
                )

    def n_gram_conv(self, n, num_filt):
        return nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = num_filt,
                kernel_size = (n, self.embed_cfg['D'])),
            nn.ReLU())

    def forward(self, h, b):
        # h = (Batch, Sentence Words)
        # b = (Batch, Sentence Words)
        h = self.h_embedding(h) # (Batch, Word, Vector)
        b = self.b_embedding(b) # (Batch, Word, Vector)
        h = h.unsqueeze(1) # (Batch, 1, Word, Vector)
        b = b.unsqueeze(1) # (Batch, 1, Word, Vector)

        # (Batch, Num_Filters, Num_Feature_Map, 1) * len(n_list)
        h_convs_out = [conv(h) for conv in self.shared_convs]

        # (Batch, Num_Filters, Num_Feature_Map, 1) * len(n_list)
        b_convs_out = [conv(b) for conv in self.shared_convs]

        # (Batch, Num_Filters, Num_Feature_Map) * len(n_list)
        h_convs_out = [output.squeeze(3) for output in h_convs_out]

        # (Batch, Num_Filters, Num_Feature_Map) * len(n_list)
        b_convs_out = [output.squeeze(3) for output in b_convs_out]

        # (Batch, Num_Filters, 1) * len(n_list)
        # MaxPool1D: 2nd arg is kernel size
        # the stride is taken to be equal to kernel size by default
        h_convs_out = [F.max_pool1d(h_conv_out, h_conv_out.shape[2])
                       for h_conv_out in h_convs_out]

        # (Batch, Num_Filters, 1) * len(n_list)
        b_convs_out = [F.max_pool1d(b_conv_out, b_conv_out.shape[2])
                       for b_conv_out in b_convs_out]

        # (Batch, Num_Filters) * len(n_list)
        h_convs_out = [h_conv_out.squeeze(2) for h_conv_out in h_convs_out]

        # (Batch, Num_Filters) * len(n_list)
        b_convs_out = [b_conv_out.squeeze(2) for b_conv_out in b_convs_out]

        # (Batch, Num_Filters * len(h_n_list))
        h_feature_vec = torch.cat(h_convs_out, dim = 1)
        b_feature_vec = torch.cat(b_convs_out, dim = 1)

        h_b_ft = torch.cat([h_feature_vec, b_feature_vec], dim = 1)
        logits = self.fc_out(h_b_ft)
        return logits, h_feature_vec, b_feature_vec

