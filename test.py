P7_U = layers.UpSampling2D()(P7_in)
P6_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
P6_td = layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                           name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
P6_U = layers.UpSampling2D()(P6_td)
P5_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U])
P5_td = layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                           name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
P5_U = layers.UpSampling2D()(P5_td)
P4_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U])
P4_td = layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                           name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
P4_U = layers.UpSampling2D()(P4_td)
P3_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
P3_out = layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                            name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
P4_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
P4_out = layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                            name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
P5_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
P5_out = layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                            name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
P6_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
P6_out = layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                            name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
P7_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
P7_out = layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                            name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)