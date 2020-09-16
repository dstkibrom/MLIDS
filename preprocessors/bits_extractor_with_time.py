import tensorflow as tf

def extract_all_bits(data):

    bit_0 = data[:, :, :1]
    bit_1 = data[:, :, 1:2]
    bit_2 = data[:, :, 2:3]
    bit_3 = data[:, :, 3:4]
    bit_4 = data[:, :, 4:5]
    bit_5 = data[:, :, 5:6]
    bit_6 = data[:, :, 6:7]
    bit_7 = data[:, :, 7:8]
    bit_8 = data[:, :, 8:9]
    bit_9 = data[:, :, 9:10]
    bit_10 = data[:, :, 10:11]
    bit_11 = data[:, :, 11:12]
    bit_12 = data[:, :, 12:13]
    bit_13 = data[:, :, 13:14]
    bit_14 = data[:, :, 14:15]
    bit_15 = data[:, :, 15:16]
    bit_16 = data[:, :, 16:17]
    bit_17 = data[:, :, 17:18]
    bit_18 = data[:, :, 18:19]
    bit_19 = data[:, :, 19:20]
    bit_20 = data[:, :, 20:21]
    bit_21 = data[:, :, 21:22]
    bit_22 = data[:, :, 22:23]
    bit_23 = data[:, :, 23:24]
    bit_24 = data[:, :, 24:25]
    bit_25 = data[:, :, 25:26]
    bit_26 = data[:, :, 26:27]
    bit_27 = data[:, :, 27:28]
    bit_28 = data[:, :, 28:29]
    bit_29 = data[:, :, 29:30]
    bit_30 = data[:, :, 30:31]
    bit_31 = data[:, :, 31:32]
    bit_32 = data[:, :, 32:33]
    bit_33 = data[:, :, 33:34]
    bit_34 = data[:, :, 34:35]
    bit_35 = data[:, :, 35:36]
    bit_36 = data[:, :, 36:37]
    bit_37 = data[:, :, 37:38]
    bit_38 = data[:, :, 38:39]
    bit_39 = data[:, :, 39:40]
    bit_40 = data[:, :, 40:41]
    bit_41 = data[:, :, 41:42]
    bit_42 = data[:, :, 42:43]
    bit_43 = data[:, :, 43:44]
    bit_44 = data[:, :, 44:45]
    bit_45 = data[:, :, 45:46]
    bit_46 = data[:, :, 46:47]
    bit_47 = data[:, :, 47:48]
    bit_48 = data[:, :, 48:49]
    bit_49 = data[:, :, 49:50]
    bit_50 = data[:, :, 50:51]
    bit_51 = data[:, :, 51:52]
    bit_52 = data[:, :, 52:53]
    bit_53 = data[:, :, 53:54]
    bit_54 = data[:, :, 54:55]
    bit_55 = data[:, :, 55:56]
    bit_56 = data[:, :, 56:57]
    bit_57 = data[:, :, 57:58]
    bit_58 = data[:, :, 58:59]
    bit_59 = data[:, :, 59:60]
    bit_60 = data[:, :, 60:61]
    bit_61 = data[:, :, 61:62]
    bit_62 = data[:, :, 62:63]
    bit_63 = data[:, :, 63:64]

    c_bit_0 = []
    for i_0 in bit_0:
        c_bit_0.append(i_0.numpy().reshape(-1).tolist())
    c_bit_1 = []
    for i_1 in bit_1:
        c_bit_1.append(i_1.numpy().reshape(-1).tolist())
    c_bit_2 = []
    for i_2 in bit_2:
        c_bit_2.append(i_2.numpy().reshape(-1).tolist())
    c_bit_3 = []
    for i_3 in bit_3:
        c_bit_3.append(i_3.numpy().reshape(-1).tolist())
    c_bit_4 = []
    for i_4 in bit_4:
        c_bit_4.append(i_4.numpy().reshape(-1).tolist())
    c_bit_5 = []
    for i_5 in bit_5:
        c_bit_5.append(i_5.numpy().reshape(-1).tolist())
    c_bit_6 = []
    for i_6 in bit_6:
        c_bit_6.append(i_6.numpy().reshape(-1).tolist())
    c_bit_7 = []
    for i_7 in bit_7:
        c_bit_7.append(i_7.numpy().reshape(-1).tolist())
    c_bit_8 = []
    for i_8 in bit_8:
        c_bit_8.append(i_8.numpy().reshape(-1).tolist())
    c_bit_9 = []
    for i_9 in bit_9:
        c_bit_9.append(i_9.numpy().reshape(-1).tolist())
    c_bit_10 = []
    for i_10 in bit_10:
        c_bit_10.append(i_10.numpy().reshape(-1).tolist())
    c_bit_11 = []
    for i_11 in bit_11:
        c_bit_11.append(i_11.numpy().reshape(-1).tolist())
    c_bit_12 = []
    for i_12 in bit_12:
        c_bit_12.append(i_12.numpy().reshape(-1).tolist())
    c_bit_13 = []
    for i_13 in bit_13:
        c_bit_13.append(i_13.numpy().reshape(-1).tolist())
    c_bit_14 = []
    for i_14 in bit_14:
        c_bit_14.append(i_14.numpy().reshape(-1).tolist())
    c_bit_15 = []
    for i_15 in bit_15:
        c_bit_15.append(i_15.numpy().reshape(-1).tolist())
    c_bit_16 = []
    for i_16 in bit_16:
        c_bit_16.append(i_16.numpy().reshape(-1).tolist())
    c_bit_17 = []
    for i_17 in bit_17:
        c_bit_17.append(i_17.numpy().reshape(-1).tolist())
    c_bit_18 = []
    for i_18 in bit_18:
        c_bit_18.append(i_18.numpy().reshape(-1).tolist())
    c_bit_19 = []
    for i_19 in bit_19:
        c_bit_19.append(i_19.numpy().reshape(-1).tolist())
    c_bit_20 = []
    for i_20 in bit_20:
        c_bit_20.append(i_20.numpy().reshape(-1).tolist())
    c_bit_21 = []
    for i_21 in bit_21:
        c_bit_21.append(i_21.numpy().reshape(-1).tolist())
    c_bit_22 = []
    for i_22 in bit_22:
        c_bit_22.append(i_22.numpy().reshape(-1).tolist())
    c_bit_23 = []
    for i_23 in bit_23:
        c_bit_23.append(i_23.numpy().reshape(-1).tolist())
    c_bit_24 = []
    for i_24 in bit_24:
        c_bit_24.append(i_24.numpy().reshape(-1).tolist())
    c_bit_25 = []
    for i_25 in bit_25:
        c_bit_25.append(i_25.numpy().reshape(-1).tolist())
    c_bit_26 = []
    for i_26 in bit_26:
        c_bit_26.append(i_26.numpy().reshape(-1).tolist())
    c_bit_27 = []
    for i_27 in bit_27:
        c_bit_27.append(i_27.numpy().reshape(-1).tolist())
    c_bit_28 = []
    for i_28 in bit_28:
        c_bit_28.append(i_28.numpy().reshape(-1).tolist())
    c_bit_29 = []
    for i_29 in bit_29:
        c_bit_29.append(i_29.numpy().reshape(-1).tolist())
    c_bit_30 = []
    for i_30 in bit_30:
        c_bit_30.append(i_30.numpy().reshape(-1).tolist())
    c_bit_31 = []
    for i_31 in bit_31:
        c_bit_31.append(i_31.numpy().reshape(-1).tolist())
    c_bit_32 = []
    for i_32 in bit_32:
        c_bit_32.append(i_32.numpy().reshape(-1).tolist())
    c_bit_33 = []
    for i_33 in bit_33:
        c_bit_33.append(i_33.numpy().reshape(-1).tolist())
    c_bit_34 = []
    for i_34 in bit_34:
        c_bit_34.append(i_34.numpy().reshape(-1).tolist())
    c_bit_35 = []
    for i_35 in bit_35:
        c_bit_35.append(i_35.numpy().reshape(-1).tolist())
    c_bit_36 = []
    for i_36 in bit_36:
        c_bit_36.append(i_36.numpy().reshape(-1).tolist())
    c_bit_37 = []
    for i_37 in bit_37:
        c_bit_37.append(i_37.numpy().reshape(-1).tolist())
    c_bit_38 = []
    for i_38 in bit_38:
        c_bit_38.append(i_38.numpy().reshape(-1).tolist())
    c_bit_39 = []
    for i_39 in bit_39:
        c_bit_39.append(i_39.numpy().reshape(-1).tolist())
    c_bit_40 = []
    for i_40 in bit_40:
        c_bit_40.append(i_40.numpy().reshape(-1).tolist())
    c_bit_41 = []
    for i_41 in bit_41:
        c_bit_41.append(i_41.numpy().reshape(-1).tolist())
    c_bit_42 = []
    for i_42 in bit_42:
        c_bit_42.append(i_42.numpy().reshape(-1).tolist())
    c_bit_43 = []
    for i_43 in bit_43:
        c_bit_43.append(i_43.numpy().reshape(-1).tolist())
    c_bit_44 = []
    for i_44 in bit_44:
        c_bit_44.append(i_44.numpy().reshape(-1).tolist())
    c_bit_45 = []
    for i_45 in bit_45:
        c_bit_45.append(i_45.numpy().reshape(-1).tolist())
    c_bit_46 = []
    for i_46 in bit_46:
        c_bit_46.append(i_46.numpy().reshape(-1).tolist())
    c_bit_47 = []
    for i_47 in bit_47:
        c_bit_47.append(i_47.numpy().reshape(-1).tolist())
    c_bit_48 = []
    for i_48 in bit_48:
        c_bit_48.append(i_48.numpy().reshape(-1).tolist())
    c_bit_49 = []
    for i_49 in bit_49:
        c_bit_49.append(i_49.numpy().reshape(-1).tolist())
    c_bit_50 = []
    for i_50 in bit_50:
        c_bit_50.append(i_50.numpy().reshape(-1).tolist())
    c_bit_51 = []
    for i_51 in bit_51:
        c_bit_51.append(i_51.numpy().reshape(-1).tolist())
    c_bit_52 = []
    for i_52 in bit_52:
        c_bit_52.append(i_52.numpy().reshape(-1).tolist())
    c_bit_53 = []
    for i_53 in bit_53:
        c_bit_53.append(i_53.numpy().reshape(-1).tolist())
    c_bit_54 = []
    for i_54 in bit_54:
        c_bit_54.append(i_54.numpy().reshape(-1).tolist())
    c_bit_55 = []
    for i_55 in bit_55:
        c_bit_55.append(i_55.numpy().reshape(-1).tolist())
    c_bit_56 = []
    for i_56 in bit_56:
        c_bit_56.append(i_56.numpy().reshape(-1).tolist())
    c_bit_57 = []
    for i_57 in bit_57:
        c_bit_57.append(i_57.numpy().reshape(-1).tolist())
    c_bit_58 = []
    for i_58 in bit_58:
        c_bit_58.append(i_58.numpy().reshape(-1).tolist())
    c_bit_59 = []
    for i_59 in bit_59:
        c_bit_59.append(i_59.numpy().reshape(-1).tolist())
    c_bit_60 = []
    for i_60 in bit_60:
        c_bit_60.append(i_60.numpy().reshape(-1).tolist())
    c_bit_61 = []
    for i_61 in bit_61:
        c_bit_61.append(i_61.numpy().reshape(-1).tolist())
    c_bit_62 = []
    for i_62 in bit_62:
        c_bit_62.append(i_62.numpy().reshape(-1).tolist())
    c_bit_63 = []
    for i_63 in bit_63:
        c_bit_63.append(i_63.numpy().reshape(-1).tolist())

    # return a ragged tensor
    return tf.ragged.constant(c_bit_0),tf.ragged.constant(c_bit_1),tf.ragged.constant(c_bit_2),tf.ragged.constant(c_bit_3),tf.ragged.constant(c_bit_4),tf.ragged.constant(c_bit_5),tf.ragged.constant(c_bit_6),tf.ragged.constant(c_bit_7),tf.ragged.constant(c_bit_8),tf.ragged.constant(c_bit_9),tf.ragged.constant(c_bit_10),tf.ragged.constant(c_bit_11),tf.ragged.constant(c_bit_12),tf.ragged.constant(c_bit_13),tf.ragged.constant(c_bit_14),tf.ragged.constant(c_bit_15),\
           tf.ragged.constant(c_bit_16),tf.ragged.constant(c_bit_17),tf.ragged.constant(c_bit_18),tf.ragged.constant(c_bit_19),tf.ragged.constant(c_bit_20),tf.ragged.constant(c_bit_21),tf.ragged.constant(c_bit_22),tf.ragged.constant(c_bit_23),tf.ragged.constant(c_bit_24),tf.ragged.constant(c_bit_25),tf.ragged.constant(c_bit_26),tf.ragged.constant(c_bit_27),tf.ragged.constant(c_bit_28),tf.ragged.constant(c_bit_29),tf.ragged.constant(c_bit_30),\
           tf.ragged.constant(c_bit_31),tf.ragged.constant(c_bit_32),tf.ragged.constant(c_bit_33),tf.ragged.constant(c_bit_34),tf.ragged.constant(c_bit_35),tf.ragged.constant(c_bit_36),tf.ragged.constant(c_bit_37),tf.ragged.constant(c_bit_38),tf.ragged.constant(c_bit_39),tf.ragged.constant(c_bit_40),tf.ragged.constant(c_bit_41),tf.ragged.constant(c_bit_42),tf.ragged.constant(c_bit_43),tf.ragged.constant(c_bit_44),tf.ragged.constant(c_bit_45),\
           tf.ragged.constant(c_bit_46),tf.ragged.constant(c_bit_47),tf.ragged.constant(c_bit_48),tf.ragged.constant(c_bit_49),tf.ragged.constant(c_bit_50),tf.ragged.constant(c_bit_51),tf.ragged.constant(c_bit_52),tf.ragged.constant(c_bit_53),tf.ragged.constant(c_bit_54),tf.ragged.constant(c_bit_55),tf.ragged.constant(c_bit_56),tf.ragged.constant(c_bit_57),tf.ragged.constant(c_bit_58),tf.ragged.constant(c_bit_59),tf.ragged.constant(c_bit_60),\
           tf.ragged.constant(c_bit_61),tf.ragged.constant(c_bit_62),tf.ragged.constant(c_bit_63)
