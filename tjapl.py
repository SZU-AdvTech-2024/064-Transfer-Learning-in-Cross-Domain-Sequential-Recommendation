import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class TJAPL(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(TJAPL, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()

        self.cross_attention_layernorms = torch.nn.ModuleList()
        self.cross_attention_layers = torch.nn.ModuleList()

        self.cross_attention_layernorms3 = torch.nn.ModuleList()
        self.cross_attention_layers3 = torch.nn.ModuleList()
        
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        # self.last_cross_layernorm = torch.nn.LayerNorm(4*args.hidden_units, eps=1e-8)
        # self.last_user_cross_layernorm = torch.nn.LayerNorm(3*args.hidden_units, eps=1e-8)
        self.last_cross_layernorm = torch.nn.LayerNorm(3 * args.hidden_units, eps=1e-8)
        self.last_user_cross_layernorm = torch.nn.LayerNorm(2*args.hidden_units, eps=1e-8)
        self.cross_forward_layernorms = torch.nn.ModuleList()
        self.cross_forward_layers = torch.nn.ModuleList()
        
        self.attention_layernorms2 = torch.nn.ModuleList()

        self.forward_layernorms2 = torch.nn.ModuleList()
        self.forward_layers2 = torch.nn.ModuleList()
        
        # self.cross_forward_layernorms3 = torch.nn.ModuleList()
        # self.cross_forward_layers3 = torch.nn.ModuleList()
        #
        # self.attention_layernorms3 = torch.nn.ModuleList()
        #
        # self.forward_layernorms3 = torch.nn.ModuleList()
        # self.forward_layers3 = torch.nn.ModuleList()
                
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)


            new_cross_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.cross_attention_layernorms.append(new_cross_attn_layernorm)

            new_attn_layernorm2 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms2.append(new_attn_layernorm2)
            
            new_cross_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.cross_attention_layers.append(new_cross_attn_layer)


            # new_cross_attn_layernorm3 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            # self.cross_attention_layernorms3.append(new_cross_attn_layernorm3)
            #
            # new_attn_layernorm3 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            # self.attention_layernorms3.append(new_attn_layernorm3)
            #
            # new_cross_attn_layer3 = torch.nn.MultiheadAttention(args.hidden_units,
            #                                                 args.num_heads,
            #                                                 args.dropout_rate)
            # self.cross_attention_layers3.append(new_cross_attn_layer3)
            
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
            
            new_fwd_layernorm2 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms2.append(new_fwd_layernorm2)

            new_fwd_layer2 = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers2.append(new_fwd_layer2)
            
            new_cross_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.cross_forward_layernorms.append(new_cross_fwd_layernorm)

            new_cross_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.cross_forward_layers.append(new_cross_fwd_layer)
            
            # new_fwd_layernorm3 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            # self.forward_layernorms3.append(new_fwd_layernorm3)
            #
            # new_fwd_layer3 = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            # self.forward_layers3.append(new_fwd_layer3)
            #
            # new_cross_fwd_layernorm3 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            # self.cross_forward_layernorms3.append(new_cross_fwd_layernorm3)
            #
            # new_cross_fwd_layer3 = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            # self.cross_forward_layers3.append(new_cross_fwd_layer3)
            
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)
        # self.gating = torch.nn.Linear(4*args.hidden_units, args.hidden_units)
        # self.gating2 = torch.nn.Linear(3*args.hidden_units, args.hidden_units)
        self.gating = torch.nn.Linear(3*args.hidden_units, args.hidden_units)
        self.gating2 = torch.nn.Linear(2*args.hidden_units, args.hidden_units)
        self.user1_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.user1_attention_layers = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
                                                            
        self.user2_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.user2_attention_layers = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
                                                            
        # self.user3_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        #
        # self.user3_attention_layers = torch.nn.MultiheadAttention(args.hidden_units,
        #                                                     args.num_heads,
        #                                                     args.dropout_rate)
        self.user_forward_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        
        self.user_forward_layers = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
                                  
        self.user_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

    # def log2user(self, user_ids, log_seqs1, log_seqs2, log_seqs3, mask2, mask3):
    def log2user(self, user_ids, log_seqs1, log_seqs2, mask2):

        seqs1 = self.item_emb(torch.LongTensor(log_seqs1).to(self.dev))
        seqs1 *= self.item_emb.embedding_dim ** 0.5
        seqs1 = self.emb_dropout(seqs1)
        timeline_mask1 = torch.BoolTensor(log_seqs1 == 0).to(self.dev)
        seqs1 *= ~timeline_mask1.unsqueeze(-1)

        tl1 = seqs1.shape[1]
        attention_mask1 = ~torch.tril(torch.ones((tl1, tl1), dtype=torch.bool, device=self.dev))
        
        user1 = self.user_emb(torch.LongTensor(user_ids).to(self.dev))
        user1 *= self.user_emb.embedding_dim ** 0.5
        user1 = self.emb_dropout(user1)
        user1 = user1.unsqueeze(1)
        user1 = user1.expand_as(seqs1)
        
        seqs1 = torch.transpose(seqs1, 0, 1)
        user1 = torch.transpose(user1, 0, 1)
        Q = self.user1_attention_layernorms(user1)
        mha_outputs1, _ = self.user1_attention_layers(Q, seqs1, seqs1, attn_mask=attention_mask1)
        user1 = mha_outputs1
        user1 = torch.transpose(user1, 0, 1)
            
            
        seqs2 = self.item_emb(torch.LongTensor(log_seqs2).to(self.dev))
        seqs2 *= self.item_emb.embedding_dim ** 0.5
        seqs2 = self.emb_dropout(seqs2)
        timeline_mask2 = torch.BoolTensor(log_seqs2 == 0).to(self.dev)
        seqs2 *= ~timeline_mask2.unsqueeze(-1)

        tl2 = seqs2.shape[1]
        attention_mask2 = mask2
        user2 = self.user_emb(torch.LongTensor(user_ids).to(self.dev))
        user2 *= self.user_emb.embedding_dim ** 0.5
        user2 = self.emb_dropout(user2)
        user2 = user2.unsqueeze(1)
        user2 = user2.expand_as(seqs2)
        
        seqs2 = torch.transpose(seqs2, 0, 1)
        user2 = torch.transpose(user2, 0, 1)
        Q2 = self.user2_attention_layernorms(user2)
        mha_outputs2, _ = self.user2_attention_layers(Q2, seqs2, seqs2, attn_mask=attention_mask2)
        user2 = mha_outputs2
        user2 = torch.transpose(user2, 0, 1)        
        
        # seqs3 = self.item_emb(torch.LongTensor(log_seqs3).to(self.dev))
        # seqs3 *= self.item_emb.embedding_dim ** 0.5
        # seqs3 = self.emb_dropout(seqs3)
        # timeline_mask3 = torch.BoolTensor(log_seqs3 == 0).to(self.dev)
        # seqs3 *= ~timeline_mask3.unsqueeze(-1)
        #
        # tl3 = seqs3.shape[1]
        # attention_mask3 = mask3
        # user3 = self.user_emb(torch.LongTensor(user_ids).to(self.dev))
        # user3 *= self.user_emb.embedding_dim ** 0.5
        # user3 = self.emb_dropout(user3)
        # user3 = user3.unsqueeze(1)
        # user3 = user3.expand_as(seqs3)
        #
        # seqs3 = torch.transpose(seqs3, 0, 1)
        # user3 = torch.transpose(user3, 0, 1)
        # Q3 = self.user3_attention_layernorms(user3)
        # mha_outputs3, _ = self.user3_attention_layers(Q3, seqs3, seqs3, attn_mask=attention_mask3)
        # user3 = mha_outputs3
        # user3 = torch.transpose(user3, 0, 1)
        
        # user = torch.cat((user1, user2, user3),dim=2)
        user = torch.cat((user1, user2), dim=2)
        user = self.last_user_cross_layernorm(user)
        user = self.dropout(self.gating2(user))

        return user


    # def log2feats(self, user_ids, log_seqs1, log_seqs2, log_seqs3, mask2, mask3):
    def log2feats(self, user_ids, log_seqs1, log_seqs2, mask2):
        seqs1 = self.item_emb(torch.LongTensor(log_seqs1).to(self.dev))
        seqs1 *= self.item_emb.embedding_dim ** 0.5
        positions1 = np.tile(np.array(range(log_seqs1.shape[1])), [log_seqs1.shape[0], 1])
        seqs1 += self.pos_emb(torch.LongTensor(positions1).to(self.dev))
        seqs1 = self.emb_dropout(seqs1)

        timeline_mask1 = torch.BoolTensor(log_seqs1 == 0).to(self.dev)
        seqs1 *= ~timeline_mask1.unsqueeze(-1)

        tl1 = seqs1.shape[1]
        attention_mask1 = ~torch.tril(torch.ones((tl1, tl1), dtype=torch.bool, device=self.dev))
        att_seq1 = seqs1

        seqs2 = self.item_emb(torch.LongTensor(log_seqs2).to(self.dev))
        seqs2 *= self.item_emb.embedding_dim ** 0.5
        positions2 = np.tile(np.array(range(log_seqs2.shape[1])), [log_seqs2.shape[0], 1])
        seqs2 += self.pos_emb(torch.LongTensor(positions2).to(self.dev))
        seqs2 = self.emb_dropout(seqs2)

        timeline_mask2 = torch.BoolTensor(log_seqs2 == 0).to(self.dev)
        seqs2 *= ~timeline_mask2.unsqueeze(-1)

        tl2 = seqs2.shape[1]
        batch_size = seqs2.shape[0]
        attention_mask2 = np.ones((batch_size, tl2, tl2), dtype=bool)
        for b in range(batch_size):
            for i in range(tl2):
                attention_mask2[b][i][0:mask2[b][i]] = False
        attention_mask2[:, :, 0] = False
        attention_mask2 = torch.from_numpy(attention_mask2).to(self.dev)
        att_seq2 = seqs1

        # seqs3 = self.item_emb(torch.LongTensor(log_seqs3).to(self.dev))
        # seqs3 *= self.item_emb.embedding_dim ** 0.5
        # positions3 = np.tile(np.array(range(log_seqs3.shape[1])), [log_seqs3.shape[0], 1])
        # seqs3 += self.pos_emb(torch.LongTensor(positions3).to(self.dev))
        # seqs3 = self.emb_dropout(seqs3)
        #
        # timeline_mask3 = torch.BoolTensor(log_seqs3 == 0).to(self.dev)
        # seqs3 *= ~timeline_mask3.unsqueeze(-1)
        #
        # tl3 = seqs3.shape[1]
        # batch_size = seqs3.shape[0]
        # attention_mask3 = np.ones((batch_size, tl3, tl3), dtype=bool)
        # for b in range(batch_size):
        #     for i in range(tl3):
        #         attention_mask3[b][i][0:mask3[b][i]] = False
        # attention_mask3[:, :, 0] = False
        # attention_mask3 = torch.from_numpy(attention_mask3).to(self.dev)
        # att_seq3 = seqs1

        for i in range(len(self.attention_layers)):
            att_seq1 = torch.transpose(att_seq1, 0, 1)

            Q = self.attention_layernorms[i](att_seq1)
            mha_outputs1, _ = self.attention_layers[i](Q, att_seq1, att_seq1,
                                                       attn_mask=attention_mask1)
            att_seq1 = Q + mha_outputs1
            att_seq1 = torch.transpose(att_seq1, 0, 1)
            
            att_seq1 = self.forward_layernorms[i](att_seq1)
            att_seq1 = self.forward_layers[i](att_seq1)
            att_seq1 *= ~timeline_mask1.unsqueeze(-1)
        seqs_2 = seqs2
        for i in range(len(self.cross_attention_layers)):
            att_seq2 = torch.transpose(att_seq2, 0, 1)
            seqs_2 = torch.transpose(seqs_2, 0, 1)

            Q = self.attention_layernorms2[i](att_seq2)
            Q2 = self.cross_attention_layernorms[i](seqs_2)
            mha_outputs2, _ = self.cross_attention_layers[i](Q, seqs_2, seqs_2,
                                                             attn_mask=attention_mask2)
            att_seq2 = Q + mha_outputs2
            
            seqs_2 = torch.transpose(seqs_2, 0, 1)
            att_seq2 = torch.transpose(att_seq2, 0, 1)
            
            att_seq2 = self.forward_layernorms2[i](att_seq2)
            att_seq2 = self.forward_layers2[i](att_seq2)
            att_seq2 *= ~timeline_mask1.unsqueeze(-1)
        # seqs_3 = seqs3
        # for i in range(len(self.cross_attention_layers3)):
        #     att_seq3 = torch.transpose(att_seq3, 0, 1)
        #     seqs_3 = torch.transpose(seqs_3, 0, 1)
        #
        #     Q = self.attention_layernorms3[i](att_seq3)
        #     Q3 = self.cross_attention_layernorms3[i](seqs_3)
        #     mha_outputs3, _ = self.cross_attention_layers3[i](Q, seqs_3, seqs_3,
        #                                              attn_mask=attention_mask3)
        #     att_seq3 = Q + mha_outputs3
        #     seqs_3 = torch.transpose(seqs_3, 0, 1)
        #     att_seq3 = torch.transpose(att_seq3, 0, 1)
        #
        #     att_seq3 = self.forward_layernorms3[i](att_seq3)
        #     att_seq3 = self.forward_layers3[i](att_seq3)
        #     att_seq3 *= ~timeline_mask1.unsqueeze(-1)
        # user_feats = self.log2user(user_ids, log_seqs1, log_seqs2, log_seqs3, attention_mask2, attention_mask3)
        # seqs = torch.cat((att_seq1, att_seq2, att_seq3, user_feats),dim=2)
        user_feats = self.log2user(user_ids, log_seqs1, log_seqs2,  attention_mask2)
        seqs = torch.cat((att_seq1, att_seq2,  user_feats), dim=2)
        seqs = self.last_cross_layernorm(seqs)
        seqs = self.dropout(self.gating(seqs))
        log_feats = self.last_layernorm(seqs)
        return log_feats
        
        
    # def forward(self, user_ids, log_seqs, log_seqs2, log_seqs3, pos_seqs, neg_seqs, mask2, mask3):
    #     log_feats = self.log2feats(user_ids, log_seqs, log_seqs2, log_seqs3, mask2, mask3)

    def forward(self, user_ids, log_seqs, log_seqs2, pos_seqs, neg_seqs, mask2):
        log_feats = self.log2feats(user_ids, log_seqs, log_seqs2, mask2)
        
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    # def predict(self, user_ids, log_seqs, log_seqs2, log_seqs3, item_indices, mask2, mask3):
    #     log_feats = self.log2feats(user_ids, log_seqs, log_seqs2, log_seqs3, mask2, mask3)

    def predict(self, user_ids, log_seqs, log_seqs2, item_indices, mask2):
        log_feats = self.log2feats(user_ids, log_seqs, log_seqs2, mask2)

        final_feat = log_feats[:, -1, :]
               
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        # item_embs=self.item_emb.weight[1:self.item_num+1,:]

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        # logits=torch.matmul(final_feat,item_embs.transpose(0,1))

        return logits
