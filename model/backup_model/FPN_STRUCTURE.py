def build_FPN(features, num_channels=64, times=0, normal_fusion=True, freeze_bn=False):

    if times == 0 :
        C3, C4, C5 = features
        P3_in = C3
        P4_in = C4
        P5_in = C5
        P6_in = Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
        # Add spatial attention
        # P6_in = BatchNormalization(freeze=freeze_bn, name='resample_p6/bn')(P6_in)
        P6_in = MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
        P6_in_2 = SA(P6_in)
        P7_in = MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
        P7_U = UpSampling2D()(P7_in)

        if normal_fusion:
            P6_td = Add(name='fpn_cells/cell_/fnode0/add')([P6_in, P7_U])
        else :
            P6_td = weightAdd(name='fpn_cells/cell_/fnode0/add')([P6_in, P7_U])

        P6_td = Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode0/op_after_combine5')(P6_td)
        P5_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name='fpn_cells/cell_/fnode1/resample_0_2_6/bn')(P5_in_1)
        # P5_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        P6_U = UpSampling2D()(P6_td)

        if normal_fusion:
            P5_td = Add(name='fpn_cells/cell_/fnode1/add')([P5_in_1, P6_U])
        else :
            P5_td = weightAdd(name='fpn_cells/cell_/fnode1/add')([P5_in_1, P6_U])

        P5_td = Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode1/op_after_combine6')(P5_td)
        P4_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode2/resample_0_1_7/conv2d')(P4_in)
        P4_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name='fpn_cells/cell_/fnode2/resample_0_1_7/bn')(P4_in_1)
        # P4_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        P5_U = UpSampling2D()(P5_td)


        if normal_fusion:
            P4_td = Add(name='fpn_cells/cell_/fnode2/add')([P4_in_1, P5_U])
        else :
            P4_td = weightAdd(name='fpn_cells/cell_/fnode2/add')([P4_in_1, P5_U])


        P4_td = Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode2/op_after_combine7')(P4_td)
        P3_in = Conv2D(num_channels, kernel_size=1, padding='same',
                              name='fpn_cells/cell_/fnode3/resample_0_0_8/conv2d')(P3_in)
        P3_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                          name=f'fpn_cells/cell_/fnode3/resample_0_0_8/bn')(P3_in)
        # P3_in = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        P4_U = UpSampling2D()(P4_td)

        if normal_fusion:
            P3_out = Add(name='fpn_cells/cell_/fnode3/add')([P3_in, P4_U])
        else :
            P3_out = weightAdd(name='fpn_cells/cell_/fnode3/add')([P3_in, P4_U])

        P3_out = Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode3/op_after_combine8')(P3_out)
        P4_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name='fpn_cells/cell_/fnode4/resample_0_1_9/bn')(P4_in_2)

        # Add spatial attention
        P4_in_2 = SA(P4_in_2)


        # P4_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)

        if normal_fusion:
            P4_out = Add(name='fpn_cells/cell_/fnode4/add')([P4_in_2, P4_td, P3_D])
        else:
            P4_out = weightAdd(name='fpn_cells/cell_/fnode4/add')([P4_in_2, P4_td, P3_D])

        P4_out = Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode4/op_after_combine9')(P4_out)

        P5_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name='fpn_cells/cell_/fnode5/resample_0_2_10/bn')(P5_in_2)

        # Add spatial attention
        P5_in_2 = SA(P5_in_2)

        # P5_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)



        if normal_fusion:
            P5_out = Add(name='fpn_cells/cell_/fnode5/add')([P5_in_2, P5_td, P4_D])
        else:
            P5_out = weightAdd(name='fpn_cells/cell_/fnode5/add')([P5_in_2, P5_td, P4_D])

        P5_out = Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode5/op_after_combine10')(P5_out)

        P5_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)

        if normal_fusion:
            P6_out = Add(name='fpn_cells/cell_/fnode6/add')([P6_in_2, P6_td, P5_D])
        else:
            P6_out = weightAdd(name='fpn_cells/cell_/fnode6/add')([P6_in_2, P6_td, P5_D])

        P6_out = Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode6/op_after_combine11')(P6_out)

        P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)

        if normal_fusion:
            P7_out = Add(name='fpn_cells/cell_/fnode7/add')([P7_in, P6_D])
        else:
            P7_out = weightAdd(name='fpn_cells/cell_/fnode7/add')([P7_in, P6_D])

        P7_out = Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode7/op_after_combine12')(P7_out)

        print('p3', P3_out)
        print('p4', P4_td)
        print('p5', P5_td)
        print('p6', P6_td)
        print('p7', P7_out)
        return P3_out, P4_td, P5_td, P6_td, P7_out

    else:
        P7_U = UpSampling2D()(P7_in)

        if normal_fusion:
            P6_td = Add(name=f'fpn_cells/cell_{times}/fnode0/add')([P6_in, P7_U])
        else:
            P6_td = weightAdd(name=f'fpn_cells/cell_{times}/fnode0/add')([P6_in, P7_U])

        P6_td = Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{times}/fnode0/op_after_combine5')(P6_td)
        P6_U = UpSampling2D()(P6_td)

        if normal_fusion:
            P5_td = Add(name=f'fpn_cells/cell_{times}/fnode1/add')([P5_in, P6_U])
        else:
            P5_td = weightAdd(name=f'fpn_cells/cell_{times}/fnode1/add')([P5_in, P6_U])

        P5_td = Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{times}/fnode1/op_after_combine6')(P5_td)
        P5_U = UpSampling2D()(P5_td)

        if normal_fusion:
            P4_td = Add(name=f'fpn_cells/cell_{times}/fnode2/add')([P4_in, P5_U])
        else:
            P4_td = weightAdd(name=f'fpn_cells/cell_{times}/fnode2/add')([P4_in, P5_U])

        P4_td = Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{times}/fnode2/op_after_combine7')(P4_td)
        P4_U = UpSampling2D()(P4_td)

        if normal_fusion:
            P3_out = Add(name=f'fpn_cells/cell_{times}/fnode3/add')([P3_in, P4_U])
        else:
            P3_out = weightAdd(name=f'fpn_cells/cell_{times}/fnode3/add')([P3_in, P4_U])

        P3_out = Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{times}/fnode3/op_after_combine8')(P3_out)
        P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)


        if normal_fusion:
            P4_out = Add(name=f'fpn_cells/cell_{times}/fnode4/add')([P4_in, P4_td, P3_D])
        else:
            P4_out = weightAdd(name=f'fpn_cells/cell_{times}/fnode4/add')([P4_in, P4_td, P3_D])

        P4_out = Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{times}/fnode4/op_after_combine9')(P4_out)

        P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)

        if normal_fusion:
            P5_out = Add(name=f'fpn_cells/cell_{times}/fnode5/add')([P5_in, P5_td, P4_D])
        else:
            P5_out = weightAdd(name=f'fpn_cells/cell_{times}/fnode5/add')([P5_in, P5_td, P4_D])

        P5_out = Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{times}/fnode5/op_after_combine10')(P5_out)

        P5_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)

        if normal_fusion:
            P6_out = Add(name=f'fpn_cells/cell_{times}/fnode6/add')([P6_in, P6_td, P5_D])
        else:
            P6_out = weightAdd(name=f'fpn_cells/cell_{times}/fnode6/add')([P6_in, P6_td, P5_D])

        P6_out = Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{times}/fnode6/op_after_combine11')(P6_out)

        P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)

        if normal_fusion:
            P7_out = Add(name=f'fpn_cells/cell_{times}/fnode7/add')([P7_in, P6_D])
        else:
            P7_out = weightAdd(name=f'fpn_cells/cell_{times}/fnode7/add')([P7_in, P6_D])

        P7_out = Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{times}/fnode7/op_after_combine12')(P7_out)

        return P3_out, P4_td, P5_td, P6_td, P7_out



    ##
    features = [conv38, conv19, conv10]
    for i in range(3):
        features = build_FPN(features, 64, times=i, normal_fusion=False)