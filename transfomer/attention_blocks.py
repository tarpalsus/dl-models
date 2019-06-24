from keras.layers import Layer, Add, Dropout, BatchNormalization
from transfomer.attention_basic_layer import MultiHeadAttention, PointwiseFeedForward
from lightconv.lightconv_layer import LightConv


class EncoderAttentionBlock(Layer):
    def __init__(self, h, embedding_size, rate=0.2, ff_dim=128, attention_type='multihead-self', k_size=7):
        super(EncoderAttentionBlock).__init__()
        self.attention_type = attention_type
        if self.attention_type == 'multihead-self':
            self.attention = MultiHeadAttention(h, embedding_size)
        elif self.attention_type == 'lightconv':
            self.attention = LightConv(k_size, h, embedding_size)
        self.position_ff = PointwiseFeedForward(ff_dim, embedding_size)
        self.rate = rate

    def __call__(self, x, mask):
        if self.attention_type == 'multihead-self':
            att = self.attention(x, x, x, mask)
        elif self.attention_type == 'lightconv':
            att = self.attention(x)
        att = Dropout(rate=self.rate)(att)
        res = Add()([att, x])
        #res = BatchNormalization(axis=0)(res)  # TODO Change to proper layer normalization, Batch norm working is a bug
        ff = self.position_ff(res)
        ff = Dropout(rate=self.rate)(ff)
        res2 = Add()([res, ff])
        #res2 = BatchNormalization(axis=0)(res2)
        return res2


class DecoderAttentionBlock(Layer):
    def __init__(self, h, embedding_size, rate=0.2, ff_dim=128, attention_type='multihead-self', k_size=7):
        super(DecoderAttentionBlock).__init__()
        self.attention_type = attention_type
        if self.attention_type == 'multihead-self':
            self.target_attention = MultiHeadAttention(h, embedding_size)
        elif self.attention_type == 'lightconv':
            self.target_attention = LightConv(k_size, h, embedding_size)
        self.input_output_attention = MultiHeadAttention(h, embedding_size)
        self.position_ff = PointwiseFeedForward(ff_dim, embedding_size)
        self.rate = rate

    def __call__(self, x, encoder_output, mask1, mask2):
        if self.attention_type == 'multihead-self':
            att = self.target_attention(x, x, x, mask1)
        elif self.attention_type == 'lightconv':
            att = self.target_attention(x)
        att = Dropout(rate=self.rate)(att)
        res = Add()([att, x])
        #res = BatchNormalization(axis=0)(res)

        att2 = self.input_output_attention(res, encoder_output, encoder_output, mask2)
        att2 = Dropout(rate=self.rate)(att2)
        res2 = Add()([att2, res])
        #res2 = BatchNormalization(axis=0)(res2)

        ff = self.position_ff(res2)
        ff = Dropout(rate=self.rate)(ff)
        res3 = Add()([res2, ff])
        #res3 = BatchNormalization(axis=0)(res3)
        return res3
