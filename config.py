MODEL_INPUT_SIZE = {
    'B0': 512,
    'B1': 576,
    'B2': 640,
    'B3': 704,
    'B4': 768,
    'B5': 832,
    'B6': 896,
    'B7': 960
}


specs = [
            Spec(int(IMAGE_SIZE[0]/16), int(IMAGE_SIZE[0]/32),
                 BoxSizes(int(IMAGE_SIZE[0]*0.1), int(IMAGE_SIZE[0]*0.24)), [2, 3]),  # 0.2
            Spec(int(IMAGE_SIZE[0]/32), int(IMAGE_SIZE[0]/16),
                 BoxSizes(int(IMAGE_SIZE[0]*0.24), int(IMAGE_SIZE[0]*0.37)), [2, 3]),  # 0.37
            Spec(int(IMAGE_SIZE[0]/64), int(IMAGE_SIZE[0]/8),
                 BoxSizes(int(IMAGE_SIZE[0]*0.45), int(IMAGE_SIZE[0]*0.58)), [2, 3]),  # 0.54
            Spec(int(IMAGE_SIZE[0]/128), int(IMAGE_SIZE[0]/4),
                 BoxSizes(int(IMAGE_SIZE[0]*0.6), int(IMAGE_SIZE[0]*0.76)), [2]),  # 0.71
            Spec(int(IMAGE_SIZE[0] / 256), int(IMAGE_SIZE[0]/2),
                 BoxSizes(int(IMAGE_SIZE[0] * 0.76), int(IMAGE_SIZE[0] * 0.9)), [2]) # 0.88 / 0.95
        ]


""" 0408 test
{'aeroplane': 0.883235057374783,
 'bicycle': 0.873452960761183,
 'bird': 0.8355587097199219,
 'boat': 0.7985908363112868,
 'bottle': 0.6096242308587395,
 'bus': 0.850391878651511,
 'car': 0.877372139861527,
 'cat': 0.873653518498321,
 'chair': 0.680718161177171,
 'cow': 0.8131513275288116,
 'diningtable': 0.7358944506308274,
 'dog': 0.8564532217146804,
 'horse': 0.8819337612866569,
 'motorbike': 0.8767863612438392,
 'person': 0.8469665526542779,
 'pottedplant': 0.6518036318412295,
 'sheep': 0.8102691940574812,
 'sofa': 0.7508058777838824,
 'train': 0.8942437806005586,
 'tvmonitor': 0.7973264964656708}
mAP결과: 0.809911607451118
"""

## test ok
# specs = [
#             Spec(64, 8, BoxSizes(51, 123), [2]),  # 0.1
#             Spec(32, 16, BoxSizes(123, 189), [2]),  # 0.24
#             Spec(16, 32, BoxSizes(189, 256), [2, 3]),  # 0.37
#             Spec(8, 64, BoxSizes(256, 323), [2, 3]),  # 0.5
#             Spec(4, 128, BoxSizes(323, 389), [2]),  # 0.63
#             Spec(2, 256, BoxSizes(389, 461), [2]),  # 0.76
#             Spec(1, 512, BoxSizes(461, 538), [2]),  # 0.9
#         ]



# specs = [
#             Spec(int(IMAGE_SIZE[0]/16), int(IMAGE_SIZE[0]/32),
#                  BoxSizes(int(IMAGE_SIZE[0]*0.1), int(IMAGE_SIZE[0]*0.24)), [2, 3]),  # 0.2
#             Spec(int(IMAGE_SIZE[0]/32), int(IMAGE_SIZE[0]/16),
#                  BoxSizes(int(IMAGE_SIZE[0]*0.24), int(IMAGE_SIZE[0]*0.37)), [2, 3]),  # 0.37
#             Spec(int(IMAGE_SIZE[0]/64), int(IMAGE_SIZE[0]/8),
#                  BoxSizes(int(IMAGE_SIZE[0]*0.5), int(IMAGE_SIZE[0]*0.63)), [2, 3]),  # 0.54
#             Spec(int(IMAGE_SIZE[0]/128), int(IMAGE_SIZE[0]/4),
#                  BoxSizes(int(IMAGE_SIZE[0]*0.63), int(IMAGE_SIZE[0]*0.76)), [2]),  # 0.71
#             Spec(int(IMAGE_SIZE[0] / 256), int(IMAGE_SIZE[0]/2),
#                  BoxSizes(int(IMAGE_SIZE[0] * 0.76), int(IMAGE_SIZE[0] * 0.9)), [2]) # 0.88 / 0.95
#         ]
