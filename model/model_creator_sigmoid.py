import tensorflow as tf


def my_model(bs_size, LSTM_units,embedding_size, num_layers):
    input_bit_0 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_0 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_0)
    if num_layers == 1:
        LSTM_layer_bit_0 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_0)
    elif num_layers == 2:
        LSTM_layer_bit_0 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_0)
        LSTM_layer_bit_0 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_0)
    else:
        LSTM_layer_bit_0 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_0)
        LSTM_layer_bit_0 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_0)
        LSTM_layer_bit_0 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_0)

    input_bit_1 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_1 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_1)
    if num_layers == 1:
        LSTM_layer_bit_1 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_1)
    elif num_layers == 2:
        LSTM_layer_bit_1 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_1)
        LSTM_layer_bit_1 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_1)
    else:
        LSTM_layer_bit_1 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_1)
        LSTM_layer_bit_1 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_1)
        LSTM_layer_bit_1 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_1)

    input_bit_2 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_2 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_2)
    if num_layers == 1:
        LSTM_layer_bit_2 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_2)
    elif num_layers == 2:
        LSTM_layer_bit_2 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_2)
        LSTM_layer_bit_2 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_2)
    else:
        LSTM_layer_bit_2 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_2)
        LSTM_layer_bit_2 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_2)
        LSTM_layer_bit_2 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_2)

    input_bit_3 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_3 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_3)
    if num_layers == 1:
        LSTM_layer_bit_3 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_3)
    elif num_layers == 2:
        LSTM_layer_bit_3 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_3)
        LSTM_layer_bit_3 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_3)
    else:
        LSTM_layer_bit_3 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_3)
        LSTM_layer_bit_3 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_3)
        LSTM_layer_bit_3 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_3)

    input_bit_4 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_4 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_4)
    if num_layers == 1:
        LSTM_layer_bit_4 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_4)
    elif num_layers == 2:
        LSTM_layer_bit_4 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_4)
        LSTM_layer_bit_4 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_4)
    else:
        LSTM_layer_bit_4 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_4)
        LSTM_layer_bit_4 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_4)
        LSTM_layer_bit_4 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_4)

    input_bit_5 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_5 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_5)
    if num_layers == 1:
        LSTM_layer_bit_5 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_5)
    elif num_layers == 2:
        LSTM_layer_bit_5 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_5)
        LSTM_layer_bit_5 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_5)
    else:
        LSTM_layer_bit_5 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_5)
        LSTM_layer_bit_5 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_5)
        LSTM_layer_bit_5 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_5)

    input_bit_6 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_6 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_6)
    if num_layers == 1:
        LSTM_layer_bit_6 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_6)
    elif num_layers == 2:
        LSTM_layer_bit_6 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_6)
        LSTM_layer_bit_6 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_6)
    else:
        LSTM_layer_bit_6 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_6)
        LSTM_layer_bit_6 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_6)
        LSTM_layer_bit_6 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_6)

    input_bit_7 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_7 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_7)
    if num_layers == 1:
        LSTM_layer_bit_7 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_7)
    elif num_layers == 2:
        LSTM_layer_bit_7 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_7)
        LSTM_layer_bit_7 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_7)
    else:
        LSTM_layer_bit_7 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_7)
        LSTM_layer_bit_7 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_7)
        LSTM_layer_bit_7 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_7)

    input_bit_8 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_8 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_8)
    if num_layers == 1:
        LSTM_layer_bit_8 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_8)
    elif num_layers == 2:
        LSTM_layer_bit_8 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_8)
        LSTM_layer_bit_8 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_8)
    else:
        LSTM_layer_bit_8 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_8)
        LSTM_layer_bit_8 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_8)
        LSTM_layer_bit_8 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_8)

    input_bit_9 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_9 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_9)
    if num_layers == 1:
        LSTM_layer_bit_9 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_9)
    elif num_layers == 2:
        LSTM_layer_bit_9 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_9)
        LSTM_layer_bit_9 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_9)
    else:
        LSTM_layer_bit_9 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_9)
        LSTM_layer_bit_9 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_9)
        LSTM_layer_bit_9 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_9)

    input_bit_10 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_10 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_10)
    if num_layers == 1:
        LSTM_layer_bit_10 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_10)
    elif num_layers == 2:
        LSTM_layer_bit_10 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_10)
        LSTM_layer_bit_10 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_10)
    else:
        LSTM_layer_bit_10 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_10)
        LSTM_layer_bit_10 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_10)
        LSTM_layer_bit_10 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_10)

    input_bit_11 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_11 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_11)
    if num_layers == 1:
        LSTM_layer_bit_11 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_11)
    elif num_layers == 2:
        LSTM_layer_bit_11 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_11)
        LSTM_layer_bit_11 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_11)
    else:
        LSTM_layer_bit_11 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_11)
        LSTM_layer_bit_11 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_11)
        LSTM_layer_bit_11 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_11)

    input_bit_12 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_12 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_12)
    if num_layers == 1:
        LSTM_layer_bit_12 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_12)
    elif num_layers == 2:
        LSTM_layer_bit_12 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_12)
        LSTM_layer_bit_12 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_12)
    else:
        LSTM_layer_bit_12 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_12)
        LSTM_layer_bit_12 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_12)
        LSTM_layer_bit_12 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_12)

    input_bit_13 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_13 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_13)
    if num_layers == 1:
        LSTM_layer_bit_13 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_13)
    elif num_layers == 2:
        LSTM_layer_bit_13 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_13)
        LSTM_layer_bit_13 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_13)
    else:
        LSTM_layer_bit_13 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_13)
        LSTM_layer_bit_13 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_13)
        LSTM_layer_bit_13 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_13)

    input_bit_14 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_14 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_14)
    if num_layers == 1:
        LSTM_layer_bit_14 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_14)
    elif num_layers == 2:
        LSTM_layer_bit_14 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_14)
        LSTM_layer_bit_14 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_14)
    else:
        LSTM_layer_bit_14 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_14)
        LSTM_layer_bit_14 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_14)
        LSTM_layer_bit_14 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_14)

    input_bit_15 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_15 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_15)
    if num_layers == 1:
        LSTM_layer_bit_15 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_15)
    elif num_layers == 2:
        LSTM_layer_bit_15 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_15)
        LSTM_layer_bit_15 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_15)
    else:
        LSTM_layer_bit_15 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_15)
        LSTM_layer_bit_15 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_15)
        LSTM_layer_bit_15 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_15)

    input_bit_16 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_16 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_16)
    if num_layers == 1:
        LSTM_layer_bit_16 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_16)
    elif num_layers == 2:
        LSTM_layer_bit_16 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_16)
        LSTM_layer_bit_16 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_16)
    else:
        LSTM_layer_bit_16 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_16)
        LSTM_layer_bit_16 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_16)
        LSTM_layer_bit_16 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_16)

    input_bit_17 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_17 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_17)
    if num_layers == 1:
        LSTM_layer_bit_17 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_17)
    elif num_layers == 2:
        LSTM_layer_bit_17 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_17)
        LSTM_layer_bit_17 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_17)
    else:
        LSTM_layer_bit_17 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_17)
        LSTM_layer_bit_17 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_17)
        LSTM_layer_bit_17 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_17)

    input_bit_18 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_18 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_18)
    if num_layers == 1:
        LSTM_layer_bit_18 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_18)
    elif num_layers == 2:
        LSTM_layer_bit_18 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_18)
        LSTM_layer_bit_18 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_18)
    else:
        LSTM_layer_bit_18 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_18)
        LSTM_layer_bit_18 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_18)
        LSTM_layer_bit_18 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_18)

    input_bit_19 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_19 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_19)
    if num_layers == 1:
        LSTM_layer_bit_19 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_19)
    elif num_layers == 2:
        LSTM_layer_bit_19 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_19)
        LSTM_layer_bit_19 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_19)
    else:
        LSTM_layer_bit_19 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_19)
        LSTM_layer_bit_19 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_19)
        LSTM_layer_bit_19 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_19)

    input_bit_20 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_20 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_20)
    if num_layers == 1:
        LSTM_layer_bit_20 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_20)
    elif num_layers == 2:
        LSTM_layer_bit_20 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_20)
        LSTM_layer_bit_20 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_20)
    else:
        LSTM_layer_bit_20 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_20)
        LSTM_layer_bit_20 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_20)
        LSTM_layer_bit_20 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_20)

    input_bit_21 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_21 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_21)
    if num_layers == 1:
        LSTM_layer_bit_21 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_21)
    elif num_layers == 2:
        LSTM_layer_bit_21 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_21)
        LSTM_layer_bit_21 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_21)
    else:
        LSTM_layer_bit_21 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_21)
        LSTM_layer_bit_21 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_21)
        LSTM_layer_bit_21 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_21)

    input_bit_22 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_22 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_22)
    if num_layers == 1:
        LSTM_layer_bit_22 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_22)
    elif num_layers == 2:
        LSTM_layer_bit_22 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_22)
        LSTM_layer_bit_22 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_22)
    else:
        LSTM_layer_bit_22 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_22)
        LSTM_layer_bit_22 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_22)
        LSTM_layer_bit_22 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_22)

    input_bit_23 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_23 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_23)
    if num_layers == 1:
        LSTM_layer_bit_23 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_23)
    elif num_layers == 2:
        LSTM_layer_bit_23 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_23)
        LSTM_layer_bit_23 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_23)
    else:
        LSTM_layer_bit_23 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_23)
        LSTM_layer_bit_23 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_23)
        LSTM_layer_bit_23 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_23)

    input_bit_24 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_24 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_24)
    if num_layers == 1:
        LSTM_layer_bit_24 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_24)
    elif num_layers == 2:
        LSTM_layer_bit_24 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_24)
        LSTM_layer_bit_24 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_24)
    else:
        LSTM_layer_bit_24 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_24)
        LSTM_layer_bit_24 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_24)
        LSTM_layer_bit_24 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_24)

    input_bit_25 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_25 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_25)
    if num_layers == 1:
        LSTM_layer_bit_25 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_25)
    elif num_layers == 2:
        LSTM_layer_bit_25 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_25)
        LSTM_layer_bit_25 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_25)
    else:
        LSTM_layer_bit_25 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_25)
        LSTM_layer_bit_25 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_25)
        LSTM_layer_bit_25 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_25)

    input_bit_26 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_26 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_26)
    if num_layers == 1:
        LSTM_layer_bit_26 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_26)
    elif num_layers == 2:
        LSTM_layer_bit_26 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_26)
        LSTM_layer_bit_26 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_26)
    else:
        LSTM_layer_bit_26 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_26)
        LSTM_layer_bit_26 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_26)
        LSTM_layer_bit_26 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_26)

    input_bit_27 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_27 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_27)
    if num_layers == 1:
        LSTM_layer_bit_27 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_27)
    elif num_layers == 2:
        LSTM_layer_bit_27 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_27)
        LSTM_layer_bit_27 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_27)
    else:
        LSTM_layer_bit_27 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_27)
        LSTM_layer_bit_27 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_27)
        LSTM_layer_bit_27 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_27)

    input_bit_28 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_28 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_28)
    if num_layers == 1:
        LSTM_layer_bit_28 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_28)
    elif num_layers == 2:
        LSTM_layer_bit_28 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_28)
        LSTM_layer_bit_28 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_28)
    else:
        LSTM_layer_bit_28 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_28)
        LSTM_layer_bit_28 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_28)
        LSTM_layer_bit_28 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_28)

    input_bit_29 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_29 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_29)
    if num_layers == 1:
        LSTM_layer_bit_29 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_29)
    elif num_layers == 2:
        LSTM_layer_bit_29 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_29)
        LSTM_layer_bit_29 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_29)
    else:
        LSTM_layer_bit_29 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_29)
        LSTM_layer_bit_29 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_29)
        LSTM_layer_bit_29 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_29)

    input_bit_30 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_30 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_30)
    if num_layers == 1:
        LSTM_layer_bit_30 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_30)
    elif num_layers == 2:
        LSTM_layer_bit_30 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_30)
        LSTM_layer_bit_30 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_30)
    else:
        LSTM_layer_bit_30 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_30)
        LSTM_layer_bit_30 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_30)
        LSTM_layer_bit_30 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_30)

    input_bit_31 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_31 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_31)
    if num_layers == 1:
        LSTM_layer_bit_31 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_31)
    elif num_layers == 2:
        LSTM_layer_bit_31 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_31)
        LSTM_layer_bit_31 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_31)
    else:
        LSTM_layer_bit_31 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_31)
        LSTM_layer_bit_31 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_31)
        LSTM_layer_bit_31 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_31)

    input_bit_32 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_32 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_32)
    if num_layers == 1:
        LSTM_layer_bit_32 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_32)
    elif num_layers == 2:
        LSTM_layer_bit_32 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_32)
        LSTM_layer_bit_32 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_32)
    else:
        LSTM_layer_bit_32 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_32)
        LSTM_layer_bit_32 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_32)
        LSTM_layer_bit_32 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_32)

    input_bit_33 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_33 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_33)
    if num_layers == 1:
        LSTM_layer_bit_33 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_33)
    elif num_layers == 2:
        LSTM_layer_bit_33 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_33)
        LSTM_layer_bit_33 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_33)
    else:
        LSTM_layer_bit_33 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_33)
        LSTM_layer_bit_33 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_33)
        LSTM_layer_bit_33 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_33)

    input_bit_34 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_34 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_34)
    if num_layers == 1:
        LSTM_layer_bit_34 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_34)
    elif num_layers == 2:
        LSTM_layer_bit_34 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_34)
        LSTM_layer_bit_34 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_34)
    else:
        LSTM_layer_bit_34 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_34)
        LSTM_layer_bit_34 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_34)
        LSTM_layer_bit_34 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_34)

    input_bit_35 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_35 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_35)
    if num_layers == 1:
        LSTM_layer_bit_35 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_35)
    elif num_layers == 2:
        LSTM_layer_bit_35 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_35)
        LSTM_layer_bit_35 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_35)
    else:
        LSTM_layer_bit_35 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_35)
        LSTM_layer_bit_35 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_35)
        LSTM_layer_bit_35 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_35)

    input_bit_36 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_36 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_36)
    if num_layers == 1:
        LSTM_layer_bit_36 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_36)
    elif num_layers == 2:
        LSTM_layer_bit_36 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_36)
        LSTM_layer_bit_36 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_36)
    else:
        LSTM_layer_bit_36 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_36)
        LSTM_layer_bit_36 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_36)
        LSTM_layer_bit_36 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_36)

    input_bit_37 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_37 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_37)
    if num_layers == 1:
        LSTM_layer_bit_37 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_37)
    elif num_layers == 2:
        LSTM_layer_bit_37 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_37)
        LSTM_layer_bit_37 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_37)
    else:
        LSTM_layer_bit_37 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_37)
        LSTM_layer_bit_37 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_37)
        LSTM_layer_bit_37 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_37)

    input_bit_38 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_38 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_38)
    if num_layers == 1:
        LSTM_layer_bit_38 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_38)
    elif num_layers == 2:
        LSTM_layer_bit_38 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_38)
        LSTM_layer_bit_38 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_38)
    else:
        LSTM_layer_bit_38 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_38)
        LSTM_layer_bit_38 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_38)
        LSTM_layer_bit_38 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_38)

    input_bit_39 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_39 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_39)
    if num_layers == 1:
        LSTM_layer_bit_39 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_39)
    elif num_layers == 2:
        LSTM_layer_bit_39 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_39)
        LSTM_layer_bit_39 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_39)
    else:
        LSTM_layer_bit_39 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_39)
        LSTM_layer_bit_39 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_39)
        LSTM_layer_bit_39 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_39)

    input_bit_40 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_40 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_40)
    if num_layers == 1:
        LSTM_layer_bit_40 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_40)
    elif num_layers == 2:
        LSTM_layer_bit_40 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_40)
        LSTM_layer_bit_40 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_40)
    else:
        LSTM_layer_bit_40 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_40)
        LSTM_layer_bit_40 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_40)
        LSTM_layer_bit_40 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_40)

    input_bit_41 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_41 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_41)
    if num_layers == 1:
        LSTM_layer_bit_41 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_41)
    elif num_layers == 2:
        LSTM_layer_bit_41 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_41)
        LSTM_layer_bit_41 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_41)
    else:
        LSTM_layer_bit_41 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_41)
        LSTM_layer_bit_41 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_41)
        LSTM_layer_bit_41 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_41)

    input_bit_42 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_42 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_42)
    if num_layers == 1:
        LSTM_layer_bit_42 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_42)
    elif num_layers == 2:
        LSTM_layer_bit_42 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_42)
        LSTM_layer_bit_42 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_42)
    else:
        LSTM_layer_bit_42 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_42)
        LSTM_layer_bit_42 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_42)
        LSTM_layer_bit_42 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_42)

    input_bit_43 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_43 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_43)
    if num_layers == 1:
        LSTM_layer_bit_43 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_43)
    elif num_layers == 2:
        LSTM_layer_bit_43 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_43)
        LSTM_layer_bit_43 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_43)
    else:
        LSTM_layer_bit_43 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_43)
        LSTM_layer_bit_43 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_43)
        LSTM_layer_bit_43 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_43)

    input_bit_44 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_44 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_44)
    if num_layers == 1:
        LSTM_layer_bit_44 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_44)
    elif num_layers == 2:
        LSTM_layer_bit_44 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_44)
        LSTM_layer_bit_44 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_44)
    else:
        LSTM_layer_bit_44 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_44)
        LSTM_layer_bit_44 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_44)
        LSTM_layer_bit_44 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_44)

    input_bit_45 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_45 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_45)
    if num_layers == 1:
        LSTM_layer_bit_45 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_45)
    elif num_layers == 2:
        LSTM_layer_bit_45 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_45)
        LSTM_layer_bit_45 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_45)
    else:
        LSTM_layer_bit_45 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_45)
        LSTM_layer_bit_45 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_45)
        LSTM_layer_bit_45 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_45)

    input_bit_46 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_46 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_46)
    if num_layers == 1:
        LSTM_layer_bit_46 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_46)
    elif num_layers == 2:
        LSTM_layer_bit_46 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_46)
        LSTM_layer_bit_46 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_46)
    else:
        LSTM_layer_bit_46 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_46)
        LSTM_layer_bit_46 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_46)
        LSTM_layer_bit_46 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_46)

    input_bit_47 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_47 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_47)
    if num_layers == 1:
        LSTM_layer_bit_47 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_47)
    elif num_layers == 2:
        LSTM_layer_bit_47 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_47)
        LSTM_layer_bit_47 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_47)
    else:
        LSTM_layer_bit_47 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_47)
        LSTM_layer_bit_47 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_47)
        LSTM_layer_bit_47 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_47)

    input_bit_48 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_48 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_48)
    if num_layers == 1:
        LSTM_layer_bit_48 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_48)
    elif num_layers == 2:
        LSTM_layer_bit_48 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_48)
        LSTM_layer_bit_48 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_48)
    else:
        LSTM_layer_bit_48 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_48)
        LSTM_layer_bit_48 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_48)
        LSTM_layer_bit_48 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_48)

    input_bit_49 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_49 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_49)
    if num_layers == 1:
        LSTM_layer_bit_49 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_49)
    elif num_layers == 2:
        LSTM_layer_bit_49 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_49)
        LSTM_layer_bit_49 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_49)
    else:
        LSTM_layer_bit_49 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_49)
        LSTM_layer_bit_49 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_49)
        LSTM_layer_bit_49 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_49)

    input_bit_50 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_50 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_50)
    if num_layers == 1:
        LSTM_layer_bit_50 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_50)
    elif num_layers == 2:
        LSTM_layer_bit_50 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_50)
        LSTM_layer_bit_50 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_50)
    else:
        LSTM_layer_bit_50 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_50)
        LSTM_layer_bit_50 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_50)
        LSTM_layer_bit_50 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_50)

    input_bit_51 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_51 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_51)
    if num_layers == 1:
        LSTM_layer_bit_51 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_51)
    elif num_layers == 2:
        LSTM_layer_bit_51 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_51)
        LSTM_layer_bit_51 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_51)
    else:
        LSTM_layer_bit_51 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_51)
        LSTM_layer_bit_51 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_51)
        LSTM_layer_bit_51 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_51)

    input_bit_52 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_52 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_52)
    if num_layers == 1:
        LSTM_layer_bit_52 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_52)
    elif num_layers == 2:
        LSTM_layer_bit_52 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_52)
        LSTM_layer_bit_52 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_52)
    else:
        LSTM_layer_bit_52 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_52)
        LSTM_layer_bit_52 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_52)
        LSTM_layer_bit_52 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_52)

    input_bit_53 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_53 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_53)
    if num_layers == 1:
        LSTM_layer_bit_53 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_53)
    elif num_layers == 2:
        LSTM_layer_bit_53 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_53)
        LSTM_layer_bit_53 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_53)
    else:
        LSTM_layer_bit_53 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_53)
        LSTM_layer_bit_53 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_53)
        LSTM_layer_bit_53 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_53)

    input_bit_54 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_54 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_54)
    if num_layers == 1:
        LSTM_layer_bit_54 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_54)
    elif num_layers == 2:
        LSTM_layer_bit_54 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_54)
        LSTM_layer_bit_54 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_54)
    else:
        LSTM_layer_bit_54 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_54)
        LSTM_layer_bit_54 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_54)
        LSTM_layer_bit_54 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_54)

    input_bit_55 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_55 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_55)
    if num_layers == 1:
        LSTM_layer_bit_55 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_55)
    elif num_layers == 2:
        LSTM_layer_bit_55 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_55)
        LSTM_layer_bit_55 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_55)
    else:
        LSTM_layer_bit_55 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_55)
        LSTM_layer_bit_55 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_55)
        LSTM_layer_bit_55 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_55)

    input_bit_56 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_56 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_56)
    if num_layers == 1:
        LSTM_layer_bit_56 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_56)
    elif num_layers == 2:
        LSTM_layer_bit_56 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_56)
        LSTM_layer_bit_56 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_56)
    else:
        LSTM_layer_bit_56 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_56)
        LSTM_layer_bit_56 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_56)
        LSTM_layer_bit_56 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_56)

    input_bit_57 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_57 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_57)
    if num_layers == 1:
        LSTM_layer_bit_57 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_57)
    elif num_layers == 2:
        LSTM_layer_bit_57 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_57)
        LSTM_layer_bit_57 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_57)
    else:
        LSTM_layer_bit_57 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_57)
        LSTM_layer_bit_57 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_57)
        LSTM_layer_bit_57 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_57)

    input_bit_58 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_58 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_58)
    if num_layers == 1:
        LSTM_layer_bit_58 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_58)
    elif num_layers == 2:
        LSTM_layer_bit_58 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_58)
        LSTM_layer_bit_58 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_58)
    else:
        LSTM_layer_bit_58 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_58)
        LSTM_layer_bit_58 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_58)
        LSTM_layer_bit_58 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_58)

    input_bit_59 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_59 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_59)
    if num_layers == 1:
        LSTM_layer_bit_59 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_59)
    elif num_layers == 2:
        LSTM_layer_bit_59 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_59)
        LSTM_layer_bit_59 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_59)
    else:
        LSTM_layer_bit_59 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_59)
        LSTM_layer_bit_59 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_59)
        LSTM_layer_bit_59 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_59)

    input_bit_60 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_60 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_60)
    if num_layers == 1:
        LSTM_layer_bit_60 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_60)
    elif num_layers == 2:
        LSTM_layer_bit_60 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_60)
        LSTM_layer_bit_60 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_60)
    else:
        LSTM_layer_bit_60 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_60)
        LSTM_layer_bit_60 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_60)
        LSTM_layer_bit_60 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_60)

    input_bit_61 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_61 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_61)
    if num_layers == 1:
        LSTM_layer_bit_61 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_61)
    elif num_layers == 2:
        LSTM_layer_bit_61 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_61)
        LSTM_layer_bit_61 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_61)
    else:
        LSTM_layer_bit_61 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_61)
        LSTM_layer_bit_61 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_61)
        LSTM_layer_bit_61 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_61)

    input_bit_62 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_62 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_62)
    if num_layers == 1:
        LSTM_layer_bit_62 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_62)
    elif num_layers == 2:
        LSTM_layer_bit_62 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_62)
        LSTM_layer_bit_62 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_62)
    else:
        LSTM_layer_bit_62 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_62)
        LSTM_layer_bit_62 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_62)
        LSTM_layer_bit_62 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_62)

    input_bit_63 = tf.keras.Input(batch_input_shape=[bs_size, None], dtype=tf.int64)
    embed_bit_63 = tf.keras.layers.Embedding(2, output_dim=embedding_size)(input_bit_63)
    if num_layers == 1:
        LSTM_layer_bit_63 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_63)
    elif num_layers == 2:
        LSTM_layer_bit_63 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_63)
        LSTM_layer_bit_63 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_63)
    else:
        LSTM_layer_bit_63 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            embed_bit_63)
        LSTM_layer_bit_63 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_63)
        LSTM_layer_bit_63 = tf.keras.layers.LSTM(LSTM_units, stateful=True, return_sequences=True, use_bias=False)(
            LSTM_layer_bit_63)

    x = tf.keras.layers.concatenate([LSTM_layer_bit_0, LSTM_layer_bit_1, LSTM_layer_bit_2, LSTM_layer_bit_3,
                                     LSTM_layer_bit_4, LSTM_layer_bit_5, LSTM_layer_bit_6, LSTM_layer_bit_7,
                                     LSTM_layer_bit_8, LSTM_layer_bit_9, LSTM_layer_bit_10, LSTM_layer_bit_11,
                                     LSTM_layer_bit_12, LSTM_layer_bit_13, LSTM_layer_bit_14, LSTM_layer_bit_15,
                                     LSTM_layer_bit_16, LSTM_layer_bit_17, LSTM_layer_bit_18, LSTM_layer_bit_19,
                                     LSTM_layer_bit_20, LSTM_layer_bit_21, LSTM_layer_bit_22, LSTM_layer_bit_23,
                                     LSTM_layer_bit_24, LSTM_layer_bit_25, LSTM_layer_bit_26, LSTM_layer_bit_27,
                                     LSTM_layer_bit_28, LSTM_layer_bit_29, LSTM_layer_bit_30, LSTM_layer_bit_31,
                                     LSTM_layer_bit_32, LSTM_layer_bit_33, LSTM_layer_bit_34, LSTM_layer_bit_35,
                                     LSTM_layer_bit_36, LSTM_layer_bit_37, LSTM_layer_bit_38, LSTM_layer_bit_39,
                                     LSTM_layer_bit_40, LSTM_layer_bit_41, LSTM_layer_bit_42, LSTM_layer_bit_43,
                                     LSTM_layer_bit_44, LSTM_layer_bit_45, LSTM_layer_bit_46, LSTM_layer_bit_47,
                                     LSTM_layer_bit_48, LSTM_layer_bit_49, LSTM_layer_bit_50, LSTM_layer_bit_51,
                                     LSTM_layer_bit_52, LSTM_layer_bit_53, LSTM_layer_bit_54, LSTM_layer_bit_55,
                                     LSTM_layer_bit_56, LSTM_layer_bit_57, LSTM_layer_bit_58, LSTM_layer_bit_59,
                                     LSTM_layer_bit_60, LSTM_layer_bit_61, LSTM_layer_bit_62, LSTM_layer_bit_63])

    Output_bit_0 = tf.keras.layers.Dense(1, activation="sigmoid")(
        x)  # no activation is used here coz using activation might not be numerically stable
    Output_bit_1 = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # from logits=True is more stable than with out it
    Output_bit_2 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_3 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_4 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_5 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_6 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_7 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_8 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_9 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_10 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_11 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_12 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_13 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_14 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_15 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_16 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_17 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_18 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_19 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_20 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_21 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_22 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_23 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_24 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_25 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_26 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_27 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_28 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_29 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_30 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_31 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_32 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_33 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_34 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_35 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_36 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_37 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_38 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_39 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_40 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_41 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_42 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_43 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_44 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_45 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_46 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_47 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_48 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_49 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_50 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_51 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_52 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_53 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_54 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_55 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_56 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_57 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_58 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_59 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_60 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_61 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_62 = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    Output_bit_63 = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[input_bit_0, input_bit_1, input_bit_2, input_bit_3, input_bit_4,
                                   input_bit_5, input_bit_6, input_bit_7, input_bit_8, input_bit_9,
                                   input_bit_10, input_bit_11, input_bit_12, input_bit_13,
                                   input_bit_14, input_bit_15, input_bit_16, input_bit_17,
                                   input_bit_18, input_bit_19, input_bit_20, input_bit_21,
                                   input_bit_22, input_bit_23, input_bit_24, input_bit_25,
                                   input_bit_26, input_bit_27, input_bit_28, input_bit_29,
                                   input_bit_30, input_bit_31, input_bit_32, input_bit_33,
                                   input_bit_34, input_bit_35, input_bit_36, input_bit_37,
                                   input_bit_38, input_bit_39, input_bit_40, input_bit_41,
                                   input_bit_42, input_bit_43, input_bit_44, input_bit_45,
                                   input_bit_46, input_bit_47, input_bit_48, input_bit_49,
                                   input_bit_50, input_bit_51, input_bit_52, input_bit_53,
                                   input_bit_54, input_bit_55, input_bit_56, input_bit_57,
                                   input_bit_58, input_bit_59, input_bit_60, input_bit_61,
                                   input_bit_62, input_bit_63],
                           outputs=[Output_bit_0, Output_bit_1, Output_bit_2, Output_bit_3,
                                    Output_bit_4, Output_bit_5, Output_bit_6, Output_bit_7,
                                    Output_bit_8, Output_bit_9, Output_bit_10, Output_bit_11,
                                    Output_bit_12, Output_bit_13, Output_bit_14, Output_bit_15,
                                    Output_bit_16, Output_bit_17, Output_bit_18, Output_bit_19,
                                    Output_bit_20, Output_bit_21, Output_bit_22, Output_bit_23,
                                    Output_bit_24, Output_bit_25, Output_bit_26, Output_bit_27,
                                    Output_bit_28, Output_bit_29, Output_bit_30, Output_bit_31,
                                    Output_bit_32, Output_bit_33, Output_bit_34, Output_bit_35,
                                    Output_bit_36, Output_bit_37, Output_bit_38, Output_bit_39,
                                    Output_bit_40, Output_bit_41, Output_bit_42, Output_bit_43,
                                    Output_bit_44, Output_bit_45, Output_bit_46, Output_bit_47,
                                    Output_bit_48, Output_bit_49, Output_bit_50, Output_bit_51,
                                    Output_bit_52, Output_bit_53, Output_bit_54, Output_bit_55,
                                    Output_bit_56, Output_bit_57, Output_bit_58, Output_bit_59,
                                    Output_bit_60, Output_bit_61, Output_bit_62, Output_bit_63])

    return model
