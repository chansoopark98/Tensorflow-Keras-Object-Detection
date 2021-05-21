import tensorflow as tf

pascal_sample_dict = {}
pascal_voc_classes_count =[ 1285,
1208,
1820,
1397,
2116,
909,
4008,
1616,
4338,
1058,
1057,
2079,
1156,
1141,
15576,
1724,
1347,
1211,
984,
1193]

avg_count = 0
for i, count in enumerate(pascal_voc_classes_count):
    pascal_sample_dict[i+1]=count
    avg_count += count

# print(pascal_sample_dict)
# print(avg_count/20)
table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], tf.float32),
        values=tf.constant([1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1 ,1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1], tf.float32),
    ),
    default_value=tf.constant(-1,tf.float32),
    name="class_weight"
)

# now let us do a lookup
input_tensor = tf.constant([0, 0, 1, 1, 2, 2, 3, 3], tf.float32)
out = table.lookup(input_tensor)
print(out)



"""
pascal voc classes count
1285
1208
1820
1397
2116
909
4008
1616
4338
1058
1057
2079
1156
1141
15576
1724
1347
1211
984
1193
"""