import sys
import numpy as np

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

POSE_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7),
    (0,4), (4,5), (5,6), (6,8),
    (9,10),
    (11,13), (13,15), (15,17), (17,19), (19,15), (15,21),
    (12,14), (14,16), (16,18), (18,20), (20,16), (16,22),
    (11,12), (12,24), (24,23), (23,11)
]

FACE_CONNECTIONS = [
    # Lips.
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
    (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
    # Left eye.
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380),
    (380, 381), (381, 382), (382, 362), (263, 466), (466, 388),
    (388, 387), (387, 386), (386, 385), (385, 384), (384, 398),
    (398, 362),
    # Left eyebrow.
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293),
    (293, 334), (334, 296), (296, 336),
    # Right eye.
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
    (153, 154), (154, 155), (155, 133), (33, 246), (246, 161),
    (161, 160), (160, 159), (159, 158), (158, 157), (157, 173),
    (173, 133),
    # Right eyebrow.
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105),
    (105, 66), (66, 107),
    # Face oval.
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
    (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
    (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
    (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
    (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109),
    (109, 10)
]

class MediaPipeVisualizer(gl.GLViewWidget) :
    
    def __init__(self) -> None :
        super().__init__()

        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-1.5, 0, 0)
        self.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -1.5, 0)
        self.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, -1.5)
        self.addItem(gz)

        self.left_hand_line_list = []
        self.right_hand_line_list = []
        self.face_line_list = []
        self.pose_line_list = []

    def updateLeftHand(self, landmark) :
        for line_item in self.left_hand_line_list :
            self.removeItem(line_item)
        landmark_list = list(map(
            lambda kp : [kp.x, kp.y, kp.z],
            landmark
        ))  
        self.left_hand_line_list = list(map(
            lambda connection : self.addItem(gl.GLLinePlotItem(
                pos = np.array([
                    landmark_list[connection[0]],
                    landmark_list[connection[1]]
                ])
            )),
            HAND_CONNECTIONS
        ))

    def updateRightHand(self, landmark) :
        for line_item in self.right_hand_line_list :
            self.removeItem(line_item)
        landmark_list = list(map(
            lambda kp : [kp.x, kp.y, kp.z],
            landmark
        ))  
        self.right_hand_line_list = list(map(
            lambda connection : self.addItem(gl.GLLinePlotItem(
                pos = np.array([
                    landmark_list[connection[0]],
                    landmark_list[connection[1]]
                ])
            )),
            HAND_CONNECTIONS
        ))

    def updatePose(self, landmark) :
        for line_item in self.pose_line_list :
            self.removeItem(line_item)
        landmark_list = list(map(
            lambda kp : [kp.x, kp.y, kp.z],
            landmark
        ))  
        self.pose_line_list = list(map(
            lambda connection : self.addItem(gl.GLLinePlotItem(
                pos = np.array([
                    landmark_list[connection[0]],
                    landmark_list[connection[1]]
                ])
            )),
            POSE_CONNECTIONS
        ))


    def updateFace(self, landmark) :
        for line_item in self.face_line_list :
            self.removeItem(line_item)
        landmark_list = list(map(
            lambda kp : [kp.x, kp.y, kp.z],
            landmark
        ))  
        self.face_line_list = list(map(
            lambda connection : self.addItem(gl.GLLinePlotItem(
                pos = np.array([
                    landmark_list[connection[0]],
                    landmark_list[connection[1]]
                ])
            )),
            POSE_CONNECTIONS
        ))

    def updateWhole(self, result) :
        self.updateLeftHand(result.left_hand_landmarks.landmark)
        self.updateRightHand(result.right_hand_landmarks.landmark)
        self.updatePose(result.pose_landmarks)
        self.updateFace(result.face_landmarks)



if __name__ == "__main__" :
    app = QtWidgets.QApplication(sys.argv)


    plot_widget = MediaPipeVisualizer()
    plot_widget.show()

    """
    [[0.665806233882904, 0.7257944941520691, 1.534599078922838e-07],
    [0.6368929147720337, 0.7211980819702148, -0.010762619785964489],
    [0.6116729974746704, 0.6878665685653687, -0.014773409813642502],
    [0.5965986847877502, 0.6560743451118469, -0.017991455271840096],
    [0.580363392829895, 0.6398572325706482, -0.02127108909189701],
    [0.6208606958389282, 0.6197853684425354, -0.007315158378332853],
    [0.6041649580001831, 0.5764697790145874, -0.014668591320514679],
    [0.593858003616333, 0.5484647750854492, -0.02111908048391342],
    [0.585757315158844, 0.5226702690124512, -0.026272648945450783],
    [0.6356361508369446, 0.6060884594917297, -0.008630681782960892],
    [0.6265880465507507, 0.5517940521240234, -0.014495390467345715],
    [0.6208407878875732, 0.5182779431343079, -0.02037525363266468],
    [0.6161496043205261, 0.48969948291778564, -0.025097601115703583],
    [0.650607168674469, 0.6052698493003845, -0.01173510029911995],
    [0.6431412100791931, 0.5547212958335876, -0.01868109591305256],
    [0.6378881931304932, 0.524552047252655, -0.023757146671414375],
    [0.6327611207962036, 0.4969666004180908, -0.027742428705096245],
    [0.6667871475219727, 0.6129403710365295, -0.015991417691111565],
    [0.6657134294509888, 0.5733337998390198, -0.0228082537651062],
    [0.6642168760299683, 0.5468123555183411, -0.02605137974023819],
    [0.6618891358375549, 0.5208349823951721, -0.028608929365873337]]
    """

    sys.exit(app.exec())


