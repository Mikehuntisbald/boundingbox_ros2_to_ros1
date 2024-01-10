import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from jsk_recognition_msgs.msg import BoundingBox

# 相机内参矩阵
K = np.array([[984.2439, 0.0, 690.0],
              [0.0, 980.8141, 233.1966],
              [0.0, 0.0, 1.0]])

# 相机畸变参数
D = np.array([-0.3728755, 0.2037299, 0.002219027, 0.001383707, -0.07233722])

# lidar 到相机的外参矩阵（旋转矩阵 R 和平移向量 T）
R = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04],
              [1.480249e-02, 7.280733e-04, -9.998902e-01],
              [9.998621e-01, 7.523790e-03, 1.480755e-02]])
T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])

class Detector: 

    def __init__(self):
        self._bridge = CvBridge()

        # 相机内参矩阵
        self.K = np.array([[984.2439, 0.0, 690.0],
                    [0.0, 980.8141, 233.1966],
                    [0.0, 0.0, 1.0]])

        # 相机畸变参数
        self.D = np.array([-0.3728755, 0.2037299, 0.002219027, 0.001383707, -0.07233722])

        # lidar 到相机的外参矩阵（旋转矩阵 R 和平移向量 T）
        self.R = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04],
                    [1.480249e-02, 7.280733e-04, -9.998902e-01],
                    [9.998621e-01, 7.523790e-03, 1.480755e-02]])
        self.T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])

        rospy.Subscriber('/kitti/camera_color_left/image_raw', Image, self.image_callback)
        rospy.Subscriber('/bounding_box_ros', BoundingBox, self.project_3d_bbox_to_2d)
        self._imagepub = rospy.Publisher('/plot_image', Image, queue_size=2)

        self._current_image = None
        
    def undistort_points(self, points_2d):
        """
        应用相机畸变校正到 2D 点上。
        :param points_2d: 2D 点的坐标。
        :param K: 相机内参矩阵。
        :param D: 相机的畸变系数。
        :return: 畸变校正后的 2D 点。
        """
        points_2d_reshaped = points_2d.T.reshape(-1, 1, 2)
        undistorted_points = cv2.undistortPoints(points_2d_reshaped, self.K, self.D, P=self.K)
        return undistorted_points.reshape(-1, 2).T
    
    def image_callback(self, image):
        """Image callback"""
        # Store value on a private attribute
        self._current_image = image


    def project_3d_bbox_to_2d(self, bbox_3d):
        """
        将 3D 边界框投影到 2D 图像平面上。
        :param K: 相机内参矩阵。
        :param D: 相机畸变系数。
        :param R: 从激光雷达到相机的旋转矩阵。
        :param T: 从激光雷达到相机的平移向量。
        :param bbox_3d: 3D 边界框的 8 个顶点。
        :return: 2D 边界框的 4 个角点 [xmin, ymin, xmax, ymax]。
        """

        if self._current_image is None:
            return
        # 构建变换矩阵
        transform = np.vstack((np.hstack((self.R, self.T[:, np.newaxis])), [0, 0, 0, 1]))

        # 将 3D 点转换为齐次坐标
        ones = np.ones((bbox_3d.shape[0], 1))
        points_homogeneous = np.hstack((bbox_3d, ones))

        # 应用变换
        points_camera = np.dot(transform, points_homogeneous.T)

        # 投影到 2D
        points_2d_homogeneous = np.dot(self.K, points_camera[:3, :])
        points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]

        # 应用畸变校正
        points_2d = self.undistort_points(points_2d)

        # 找到 2D 边界框
        xmin = np.min(points_2d[0, :])
        ymin = np.min(points_2d[1, :])
        xmax = np.max(points_2d[0, :])
        ymax = np.max(points_2d[1, :])

        # 将边界框坐标转换为整数
        bbox_2d = np.array([xmin, ymin, xmax, ymax], dtype=int)

        cv_image = self._bridge.imgmsg_to_cv2(self._current_image, 'bgr8')
        # 绘制边界框
        cv2.rectangle(cv_image, (bbox_2d[0], bbox_2d[1]), (bbox_2d[2], bbox_2d[3]), (0, 255, 0), 2)
        
        self._imagepub.publish(self._bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
        # return plot_image
    
if __name__ == '__main__':

    # ROS 节点初始化
    rospy.init_node('point_cloud_to_2d', anonymous=True)
    Detector()
    rospy.spin()

    # bbox_3d = np.array(...)  # 3D 边界框的 8 个顶点

    # bbox_2d = project_3d_bbox_to_2d(K, D, R, T, bbox_3d)

