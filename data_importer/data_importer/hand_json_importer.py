# next(iter(data_json))
import json
import numpy as np
# data_json=[json.lao]
# anyjson=str(any_one(data_json))
anyjson=str(next(it))
dt=json.load(open(anyjson))

arr_left = np.array(dt['people'][0]['hand_left_keypoints_2d']).reshape(21,3)
arr_right = np.array(dt['people'][0]['hand_right_keypoints_2d']).reshape(21,3)

# draw_point(ay.data.gtorig,ay.hpe)
draw_point(arr_left,arr_right,ay.hpe)