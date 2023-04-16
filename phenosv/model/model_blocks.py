import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import math
import model.operation_function as of

#reference:  https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
def scaled_dot_product(q, k, v, mask=None,sk=None, sv=None):
    #mask shape: batch, head , SeqLen SeqLen, outerproduct of gene indicator
    #give value as 0 for non-attentions, usuful for zero paddings
    #q, l, v: batch, head, SeqLen, emb
    #sk, sv: seqlen, seqlen, emb
    batch_size, nhead, seq_len, d_k=k.size()
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    if sk is not None:
        q_ = q.reshape(batch_size*nhead, seq_len, d_k).permute(1,0,2)
        attn_logits1 = torch.matmul(q_,sk.transpose(1,2)).transpose(0,1) #emb, seq, seq
        attn_logits1 = attn_logits1.reshape(batch_size, nhead,seq_len,seq_len)
        attn_logits = attn_logits+attn_logits1
    attn_logits = attn_logits / math.sqrt(d_k)#batch, head , SeqLen SeqLen
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    sm = nn.Softmax(dim=-1)
    attention = sm(attn_logits)#row sum equals to 1 #batch head seq seq
    values = torch.matmul(attention, v)#batch head seq emb
    if sv is not None:
        attention_ = attention.permute(2,0,1,3).reshape(seq_len, batch_size * nhead, seq_len)
        values1 = torch.matmul(attention_, sv) #seq_len, batch_size * nhead, d_k
        values1 = values1.transpose(0,1).reshape(batch_size, nhead,seq_len,d_k)
        values = values+values1

    return values, attention

def batch_mask(mask1,mask2):
    return torch.bmm(mask1[:, :, 0].unsqueeze(2), mask2[:, :, 0].unsqueeze(1))

class RelativePositionalEncoding(nn.Module):
    #https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
    #out: [seq, seq, emb]
    def __init__(self, d, max_relative_position, clip_pad=True):
        #max_relative_position is not max len, max_len = 2*max_relative_position+1
        super().__init__()
        self.d = d
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, d))
        self.clip_pad=clip_pad
        nn.init.xavier_uniform_(self.embeddings_table)
    def forward(self, length_q, length_k): #seq_len; seq_len
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        if self.clip_pad:
            distance_mat_clipped[1:-1, 0] = -self.max_relative_position
            distance_mat_clipped[1:-1, -1] = self.max_relative_position
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)#.cuda()
        embeddings = self.embeddings_table[final_mat]#.cuda()
        return embeddings

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
        d_model - Hidden dimensionality of the input.
        max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, max_relative_position=0, clip_pad = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.clip_pad = clip_pad
        self.max_relative_position = max_relative_position
        if max_relative_position>0:
            self.relative_position_k = RelativePositionalEncoding(self.head_dim, self.max_relative_position,self.clip_pad)
            self.relative_position_v = RelativePositionalEncoding(self.head_dim, self.max_relative_position,self.clip_pad)

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)#QW, KW, VW
        self.o_proj = nn.Linear(embed_dim, embed_dim)#original

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims( dim of 3 W matrix of QKV)]
        q, k, v = qkv.chunk(3, dim=-1)# each with size: batch, head, SeqLen emb

        #add relative positional encoding onto k and v
        if self.max_relative_position > 0:
            sk = self.relative_position_k(seq_length, seq_length)
            sv = self.relative_position_k(seq_length, seq_length)
            values, attention = scaled_dot_product(q, k, v, mask=mask, sk= sk, sv=sv)
        else:
            # Determine value outputs
            values, attention = scaled_dot_product(q, k, v, mask=mask)


        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim) #head * dim
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0, max_relative_position=0,clip_pad=True):
        """
        Inputs:
        input_dim - Dimensionality of the input
        num_heads - Number of heads to use in the attention block
        dim_feedforward - Dimensionality of the hidden layer in the MLP
        dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads, max_relative_position,clip_pad)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        # Attention part
        attn_out = self.self_attn(x, mask= mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

class MIL(nn.Module):
    def __init__(self, input_dim, embed_dim):
        """
        Inputs:
        input_dim - Dimensionality of the input
        embed_dim - Dimensionality of the hidden layer for attention, A
        """
        super().__init__()
        # Attention layer
        self.attention = nn.Sequential(
                         nn.Linear(input_dim, embed_dim),
                         nn.Tanh(),
                         nn.Linear(embed_dim, 1),
                         nn.Sigmoid()
        )
    def forward(self, x, gene_indicators=None):
        # Attention part
        element_weights = self.attention(x)#batch * seq *1
        if gene_indicators is not None:
            element_weights = torch.squeeze(element_weights,dim=-1) * gene_indicators  # batch seq
        else:
            element_weights = torch.squeeze(element_weights,dim=-1)
        return element_weights

class MIL2(nn.Module):
    def __init__(self, input_dim, embed_dim):
        """
        Inputs:
        input_dim - Dimensionality of the input
        embed_dim - Dimensionality of the hidden layer for attention, A, if 0 use max-pooling
        """
        super().__init__()
        # Attention layer
        self.attention = nn.Sequential(
                         nn.Linear(input_dim, embed_dim),
                         nn.Tanh(),
                         nn.Linear(embed_dim, 1))
        self.sm = nn.Softmax(dim=-2)
    def forward(self, x):
        # Attention part
        A = self.attention(x)#batch * seq *1
        A = self.sm(A)
        A = torch.squeeze(A,dim = -1)#batch * seq *1 --> batch*seq
        return A

class LocalFeatureExtractor(nn.Module):
    def __init__(self, input_dim, input_middle_dim, model_dim, num_heads, number_layers_encoder,  dropout=0.0,
                 sentence_embedding=False, max_relative_position=0, clip_pad=True):
        """
        Inputs:
        input_dim - original dimensionality of the input (element feature dimension, 282)
        input_middle_dim - Hidden dimensionality to use before feed into transformer
        model_dim - Hidden dimensionality to use inside the Transformer (256)
        num_heads - Number of heads to use in the Multi-Head Attention blocks (4)
        number_layers_encoder - Number of encoder blocks to use, attention between sequences. (8)
        max_relative_position -- max relative len, set to a positive number if using relative positional encoding
        dropout - Dropout to apply inside the model (0.25)
        """
        super().__init__()
        self.input_dim = input_dim
        self.input_middle_dim = input_middle_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.number_layers_encoder = number_layers_encoder
        self.dropout = dropout
        self.sentence_embedding=sentence_embedding
        self.max_relative_position = max_relative_position
        self.clip_pad = clip_pad

        self.input_net = nn.Sequential(
                nn.Linear(input_dim, input_middle_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(input_middle_dim, model_dim))

        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=model_dim)

        # Transformer encoder --self
        self.encoder = TransformerEncoder(num_layers=number_layers_encoder,
                                          input_dim=model_dim,
                                          dim_feedforward=2 * model_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          max_relative_position = self.max_relative_position,
                                          clip_pad = self.clip_pad)

    def forward(self, x,  padding_mask, gene_indicators, sv_indicators):
        #sv_indicator: used for global model: gene attend themself and noncoding sv; if None, train coding local model
        x = self.input_net(x)
        # add absolute positional encoding if not using relative positional encoding
        if self.max_relative_position==0:
            x = self.positional_encoding(x)
        # self attention encoding, add padding masks of shape: [batch, seq, model_embed]
        mask = batch_mask(padding_mask,padding_mask).unsqueeze(1).repeat(1, self.num_heads, 1, 1)#batch head seq seq
        indicators_mat = of.generate_attention_mask(gene_indicators, sv_indicators, self.num_heads)
        mask = mask * indicators_mat
        if self.sentence_embedding:
            mask[:, :, 0, :] = 1.0
        instance_embedding = self.encoder(x, mask=mask)
        return instance_embedding

    def get_attention_maps(self, x, gene_indicators, sv_indicators):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        type: choose from 'encoder' or 'crossencoder'
        Can put padding mask
        """
        x = self.input_net(x)
        if self.max_relative_position == 0:
            x = self.positional_encoding(x)
        mask = of.generate_attention_mask(gene_indicators, sv_indicators, self.num_heads)
        if self.sentence_embedding:
            mask[:, :, 0, :] = 1.0
        attention_maps = self.encoder.get_attention_maps(x, mask=mask)
        return attention_maps

class ClassificationLayer(nn.Module):
    def __init__(self, input_dim, middel_dim, attention_dim):

        super().__init__()
        # set to >0: attention ; 0: max-pooling; -1: sentance embedding;
        # -2: use prediction back prop;
        self.attention_dim = attention_dim
        self.classifier = nn.Sequential(
                 nn.Linear(input_dim, middel_dim),
                 nn.ReLU(),
                 nn.Linear(middel_dim, 1)
        )#output logit


        # MIL gene attention
        if self.attention_dim > 0:
            self.attention = MIL2(input_dim=input_dim, embed_dim=attention_dim)
        else:
            self.attention = None

    def forward(self, gene_embedding,  sv_indicators = None, grad_reverse=False):
        #sv_indicators: pooling indicators
        gene_weights=None
        if self.attention_dim == 0: #max-pooling
            if sv_indicators is not None: #max-pooling sv region
                sv_indicators = sv_indicators.unsqueeze(-1).repeat(1, 1, gene_embedding.size()[-1])#batch seq emb
                sv_indicators = (1 - sv_indicators).bool()
                gene_embedding[sv_indicators] = -99
                x = torch.max(gene_embedding, dim=-2, keepdim=True)[0]#batch seq 1-->batch seq emb
            else: #max-pooling all
                x = torch.max(gene_embedding, dim=-2, keepdim=True)[0]  # batch seq 1-->batch seq emb
            embedding = torch.squeeze(x)

        elif self.attention_dim==-1: #sentence embedding
            embedding = gene_embedding[:,0,:]

        elif self.attention_dim>0:#attention machenism
            gene_weights = self.attention(gene_embedding)
            if sv_indicators is not None:
                gene_weights = gene_weights*sv_indicators
            gene_weights = torch.unsqueeze(gene_weights, 1)  # batch 1 seq
            embedding = torch.matmul(gene_weights.float(), gene_embedding)
        elif self.attention_dim==-4:#average pooling
            if sv_indicators is not None: #average-pooling sv region
                sv_indicators = sv_indicators.unsqueeze(-1).repeat(1, 1, gene_embedding.size()[-1])#batch seq emb
                gene_embedding=gene_embedding*sv_indicators
                x = torch.sum(gene_embedding, dim=1, keepdim=True)/ torch.sum(sv_indicators, dim=1, keepdim=True)#batch seq 1-->batch seq emb
            else: #average-pooling all
                x = torch.mean(gene_embedding, dim=1, keepdim=True)  # batch seq 1-->batch seq emb
            embedding = torch.squeeze(x)
        else:  #-2,do not aggregate, use max prediction to BP; -3, do not aggregate, instance learning
            embedding = gene_embedding

        embedding = embedding.float()
        if grad_reverse:
            embedding=of.GradReverse.apply(embedding)

        y_pred = self.classifier(embedding)
        if self.attention_dim == -2:
            if sv_indicators is not None:
                sv_indicators = sv_indicators.unsqueeze(-1)
                sv_indicators = (1 - sv_indicators).bool()
                y_pred[sv_indicators] = -999
                y_pred = torch.max(y_pred, dim=1, keepdim=True)[0]#batch seq 1-->batch seq emd
            else:
                y_pred = torch.max(y_pred,dim=1,keepdim=True)[0]
        y_pred = y_pred.squeeze()
        if self.attention_dim == -3 and len(y_pred.size())==1:#instance learning
            y_pred = y_pred.unsqueeze(0)

        return y_pred, gene_weights #output logit

