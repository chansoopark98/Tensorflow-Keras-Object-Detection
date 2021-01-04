from matplotlib import pyplot as plt

# x축은 parameters
x = ['B0','B1','B2','B3','B4','B5','B6','B7']
# y축은 mAP
y = [5.3, 7.8, 9.2, 12, 19, 30, 43, 66]
# 여기서 그리기 옵션 조정하시면 됩니다
plt.plot(x ,y ,color='blue')
# plt.plot(x[1] ,y[1], '*',markersize=15,color='blue')
# plt.plot(x[2] ,y[2], '^', markersize=15,color='red')
#plt.xlim([0,100])
#plt.ylim([70,90])
#plt.legend(['SSD', 'YOLOv2', 'Proposed'])
plt.xlabel('EfficientNet Model Name')
plt.ylabel('Model Parameters (Million)')
plt.grid(True)

plt.show()
