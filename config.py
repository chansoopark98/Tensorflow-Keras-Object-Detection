from utils.priors import *

iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2


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
def set_priorBox(model_name):
        return [
            Spec(32, 16, BoxSizes(51, 122), [2,3]),
            Spec(16, 32, BoxSizes(122, 189), [2,3]),
            Spec(8, 64, BoxSizes(230, 296), [2,3]),
            Spec(4, 128, BoxSizes(307, 389), [2]),
            Spec(2, 256, BoxSizes(389, 460), [2]),
        ]



# def set_priorBox(model_name):
#     IMAGE_SIZE = MODEL_INPUT_SIZE[model_name]
#     if model_name == 'B0':
#
#         return [
#             Spec(32, 16, BoxSizes(51, 122), [2,3]),
#             Spec(16, 32, BoxSizes(122, 189), [2,3]),
#             Spec(8, 64, BoxSizes(230, 296), [2,3]),
#             Spec(4, 128, BoxSizes(307, 389), [2]),
#             Spec(2, 256, BoxSizes(389, 460), [2]),
#         ]
#
#     elif model_name == 'B1':
#
#         return [
#             Spec(36, 18,  BoxSizes(int(IMAGE_SIZE*0.1), int(IMAGE_SIZE*0.24)), [2, 3]),
#             Spec(18, 36, BoxSizes(int(IMAGE_SIZE*0.24), int(IMAGE_SIZE*0.37)), [2, 3]),
#             Spec(9, 72, BoxSizes(int(IMAGE_SIZE*0.45), int(IMAGE_SIZE*0.58)), [2, 3]),
#             Spec(4, 144, BoxSizes(int(IMAGE_SIZE*0.6), int(IMAGE_SIZE*0.76)), [2]),
#             Spec(2, 288, BoxSizes(int(IMAGE_SIZE * 0.76), int(IMAGE_SIZE * 0.9)), [2])
#         ]
#     elif model_name == 'B2':
#
#         return [
#             Spec(40, 20,  BoxSizes(int(IMAGE_SIZE*0.1), int(IMAGE_SIZE*0.24)), [2, 3]),
#             Spec(20, 40, BoxSizes(int(IMAGE_SIZE*0.24), int(IMAGE_SIZE*0.37)), [2, 3]),
#             Spec(10, 80, BoxSizes(int(IMAGE_SIZE*0.45), int(IMAGE_SIZE*0.58)), [2, 3]),
#             Spec(4, 160, BoxSizes(int(IMAGE_SIZE*0.6), int(IMAGE_SIZE*0.76)), [2]),
#             Spec(2, 320, BoxSizes(int(IMAGE_SIZE * 0.76), int(IMAGE_SIZE * 0.9)), [2])
#         ]
#
#     elif model_name == 'B3':
#
#         return [
#             Spec(44, 22,  BoxSizes(int(IMAGE_SIZE*0.1), int(IMAGE_SIZE*0.24)), [2, 3]),
#             Spec(22, 44, BoxSizes(int(IMAGE_SIZE*0.24), int(IMAGE_SIZE*0.37)), [2, 3]),
#             Spec(11, 88, BoxSizes(int(IMAGE_SIZE*0.45), int(IMAGE_SIZE*0.58)), [2, 3]),
#             Spec(5, 176, BoxSizes(int(IMAGE_SIZE*0.6), int(IMAGE_SIZE*0.76)), [2]),
#             Spec(3, 352, BoxSizes(int(IMAGE_SIZE * 0.76), int(IMAGE_SIZE * 0.9)), [2])
#         ]

# specs = [
#             Spec(int(IMAGE_SIZE[0]/16), int(IMAGE_SIZE[0]/32),
#                  BoxSizes(int(IMAGE_SIZE[0]*0.1), int(IMAGE_SIZE[0]*0.24)), [2, 3]),  # 0.2
#             Spec(int(IMAGE_SIZE[0]/32), int(IMAGE_SIZE[0]/16),
#                  BoxSizes(int(IMAGE_SIZE[0]*0.24), int(IMAGE_SIZE[0]*0.37)), [2, 3]),  # 0.37
#             Spec(int(IMAGE_SIZE[0]/64), int(IMAGE_SIZE[0]/8),
#                  BoxSizes(int(IMAGE_SIZE[0]*0.45), int(IMAGE_SIZE[0]*0.58)), [2, 3]),  # 0.54
#             Spec(int(IMAGE_SIZE[0]/128), int(IMAGE_SIZE[0]/4),
#                  BoxSizes(int(IMAGE_SIZE[0]*0.6), int(IMAGE_SIZE[0]*0.76)), [2]),  # 0.71
#             Spec(int(IMAGE_SIZE[0] / 256), int(IMAGE_SIZE[0]/2),
#                  BoxSizes(int(IMAGE_SIZE[0] * 0.76), int(IMAGE_SIZE[0] * 0.9)), [2]) # 0.88 / 0.95
#         ]


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

"""
0409_b0_model

AP 결과
{'aeroplane': 0.8720887268352158,
 'bicycle': 0.8852008080603738,
 'bird': 0.8256423371316157,
 'boat': 0.7736810588067703,
 'bottle': 0.6161642842571923,
 'bus': 0.838851283409325,
 'car': 0.8755407635821656,
 'cat': 0.8825923751396492,
 'chair': 0.6883221404468115,
 'cow': 0.8666497816152792,
 'diningtable': 0.7631169047748221,
 'dog': 0.8654741294935093,
 'horse': 0.8811329388837018,
 'motorbike': 0.8800345134156637,
 'person': 0.8481771382721582,
 'pottedplant': 0.6534823656746408,
 'sheep': 0.8112723123853273,
 'sofa': 0.7938674460698647,
 'train': 0.8893720774945586,
 'tvmonitor': 0.7781294727508264}
mAP결과: 0.8144396429249735


Total params: 4,029,830
Trainable params: 3,986,406
Non-trainable params: 43,424

FLOPS : 4074691264 (4.07B)
"""

"""
0410_b1_model
AP 결과
{'aeroplane': 0.8828147181765603,
 'bicycle': 0.8861097958992193,
 'bird': 0.8221666165500555,
 'boat': 0.7894837235067005,
 'bottle': 0.6489816559206699,
 'bus': 0.8500586612504307,
 'car': 0.8847536066022171,
 'cat': 0.875707964961166,
 'chair': 0.6997916153092536,
 'cow': 0.8418244882156247,
 'diningtable': 0.766302726432726,
 'dog': 0.8712808876453744,
 'horse': 0.8850078665050914,
 'motorbike': 0.8747021135077606,
 'person': 0.8621754981732161,
 'pottedplant': 0.6433761230393479,
 'sheep': 0.8311298049426942,
 'sofa': 0.7817974028171251,
 'train': 0.8990094002356919,
 'tvmonitor': 0.7906922595050226}
mAP결과: 0.8193583464597974

Total params: 6,778,266
Trainable params: 6,711,914
Non-trainable params: 66,352

FLOPS : 6112267328 (6.1B)

"""
