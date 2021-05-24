# import cv2

# START_TIME= 0 #设置开始时间(单位秒)
# END_TIME= 73 #设置结束时间(单位秒)

# vidcap = cv2.VideoCapture(r'D:\QATM_pytorch\视频\1.mp4')

# fps = int(vidcap.get(cv2.CAP_PROP_FPS))  # 获取视频每秒的帧数
# print(fps)

# frameToStart = START_TIME*fps #开始帧 = 开始时间*帧率
# print(frameToStart)
# frametoStop = END_TIME*fps #结束帧 = 结束时间*帧率
# print(frametoStop)

# vidcap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart) #设置读取的位置,从第几帧开始读取视频
# print(vidcap.get(cv2.CAP_PROP_POS_FRAMES))  # 查看当前的帧数

# success,image = vidcap.read()  # 获取第一帧

# count = 0
# while success and frametoStop >= count:
#     if count % (10) == 0:  # 每10帧（1/3秒）保存一次
#         cv2.imwrite(r'D:\QATM_pytorch\视频截图\%d.jpg' % int(count / 10), image)  # 保存图片
#         print('Process %dth seconds: ' % int(count / 10), success)
#     success,image = vidcap.read()  # 每次读取一帧
#     count += 1

# print("end!")
import cv2
 
def video2frame(videos_path,frames_save_path,time_interval):
 
  '''
  :param videos_path: 视频的存放路径
  :param frames_save_path: 视频切分成帧之后图片的保存路径
  :param time_interval: 保存间隔
  :return:
  '''
  vidcap = cv2.VideoCapture(videos_path)
  success, image = vidcap.read()
  count = 0
  print(videos_path)
  while success:
    success, image = vidcap.read()
    count += 1
    if count % time_interval == 0:
      cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "/frame%d.jpg" % count)
    # if count == 20:
    #   break
  print(count)
 
if __name__ == '__main__':
   videos_path = 'D:\QATM_pytorch\视频\\2.mp4'
   frames_save_path = 'D:\QATM_pytorch\视频截图2'
   time_interval = 10#隔一帧保存一次
   video2frame(videos_path, frames_save_path, time_interval)