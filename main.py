import cv2
import keyboard
import numpy as np
import dlib

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_FPS, 10)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_data.dat")


def get_index(array):
	index = None

	for i in array[0]:
		index = i
		break

	return index


def find_landmarks(image):
	points = []

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	faces = detector(gray)
	for face in faces:
		landmarks = predictor(gray, face)


		for i in range(0, 68):
			x = landmarks.part(i).x
			y = landmarks.part(i).y

			points.append([x, y])
			cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

	return points


def deepfake(image, target, final):
	cropped1 = None
	cropped2 = None


	points = find_landmarks(image)
	points_numpy_arr = np.array(points, np.int32)

	target_points = find_landmarks(target)
	target_numpy_arr = np.array(target_points, np.int32)

	if (target_points != None):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		convex_hull = cv2.convexHull(points_numpy_arr)
		#cv2.polylines(image, [convex_hull], True, (255, 0, 0), 3)

		target_convex_hull = cv2.convexHull(target_numpy_arr)

		mask = np.zeros_like(gray)
		cv2.fillConvexPoly(mask, convex_hull, 255)

		face_1 = cv2.bitwise_and(image, image, mask=mask)

		rectangle = cv2.boundingRect(convex_hull)
		sub_div = cv2.Subdiv2D(rectangle)
		sub_div.insert(points)
		triangles = np.array(sub_div.getTriangleList(), np.int32)


		triangle_indexes = []
		for triangle in triangles:
			point_1 = [triangle[0], triangle[1]]
			point_2 = [triangle[2], triangle[3]]
			point_3 = [triangle[4], triangle[5]]

			#cv2.line(image, point_1, point_2, (255, 0, 255), 2)
			#cv2.line(image, point_2, point_3, (255, 0, 255), 2)
			#cv2.line(image, point_3, point_1, (255, 0, 255), 2)

			
			point_1_index = get_index(np.where((points_numpy_arr == point_1).all(axis = 1)))
			point_2_index = get_index(np.where((points_numpy_arr == point_2).all(axis = 1)))
			point_3_index = get_index(np.where((points_numpy_arr == point_3).all(axis = 1)))
			

			if (point_1_index != None) & (point_2_index != None) & (point_3_index != None):
				triangle_complete = [point_1_index, point_2_index, point_3_index]
				triangle_indexes.append(triangle_complete)


		for triangle_index in triangle_indexes:
			image_point_1 = points[triangle_index[0]]
			image_point_2 = points[triangle_index[1]]
			image_point_3 = points[triangle_index[2]]










			image_triangle = np.array([image_point_1, image_point_2, image_point_3], np.int32)
			(x, y, width, height) = cv2.boundingRect(image_triangle)
			cropped1 = image[y: y + height, x: x + width]
			
			cropped1_mask = np.zeros((height, width), np.uint8)
			image_final_points = np.array([[image_point_1[0] - x, image_point_1[1] - y],
										  [image_point_2[0] - x, image_point_2[1] - y],
										  [image_point_3[0] - x, image_point_3[1] - y]], np.int32)
			
			cv2.fillConvexPoly(cropped1_mask, image_final_points, 255)
			cropped1 = cv2.bitwise_and(cropped1, cropped1, mask=cropped1_mask)











			#cv2.line(image, image_point_1, image_point_2, (0, 0, 255), 2)
			#cv2.line(image, image_point_2, image_point_3, (0, 0, 255), 2)
			#cv2.line(image, image_point_3, image_point_1, (0, 0, 255), 2)


			target_point_1 = target_points[triangle_index[0]]
			target_point_2 = target_points[triangle_index[1]]
			target_point_3 = target_points[triangle_index[2]]

			target_triangle = np.array([target_point_1, target_point_2, target_point_3], np.int32)
			(x, y, width, height) = cv2.boundingRect(target_triangle)
			cropped2 = target[y: y + height, x: x + width]
			
			cropped2_mask = np.zeros((height, width), np.uint8)
			target_final_points = np.array([[target_point_1[0] - x, target_point_1[1] - y],
										  [target_point_2[0] - x, target_point_2[1] - y],
										  [target_point_3[0] - x, target_point_3[1] - y]], np.int32)
			
			cv2.fillConvexPoly(cropped2_mask, target_final_points, 255)
			cropped2 = cv2.bitwise_and(cropped2, cropped2, mask=cropped2_mask)









			#cv2.line(target, target_point_1, target_point_2, (0, 0, 255), 2)
			#cv2.line(target, target_point_2, target_point_3, (0, 0, 255), 2)
			#cv2.line(target, target_point_3, target_point_1, (0, 0, 255), 2)






			image_final_points = np.float32(image_final_points)
			target_final_points = np.float32(target_final_points)

			warp_data = cv2.getAffineTransform(image_final_points, target_final_points)
			warped_triangle = cv2.warpAffine(cropped1, warp_data, (width, height), flags=cv2.INTER_NEAREST)

			
			triangle_area = final[y: y + height, x: x + width]
			triangle_area = cv2.add(triangle_area, warped_triangle)
			final[y: y + height, x: x + width] = triangle_area

			target_face_mask = np.zeros_like(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY))
			target_head_mask = cv2.fillConvexPoly(target_face_mask, target_convex_hull, 255)
			target_face_mask = cv2.bitwise_not(target_head_mask)

			target_head = cv2.bitwise_and(target, target, mask=target_face_mask)
			combined = cv2.add(target_head, final)

			(x, y, width, height) = cv2.boundingRect(target_convex_hull)
			target_center_face = (int((x + x + width) / 2), int((y + y + height) / 2))
			deepfake_clone = cv2.seamlessClone(combined, target, target_head_mask, target_center_face, cv2.MIXED_CLONE)
			deepfake_blur = cv2.medianBlur(deepfake_clone, 5)
			sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
			deepfake_sharp = cv2.filter2D(deepfake_blur, -1, sharpen_kernel)


		return deepfake_blur


zuck = cv2.resize(cv2.imread("zuck.jpg"), (0, 0), fx=0.25, fy=0.25)
elon = cv2.resize(cv2.imread("elon.jpg"), (0, 0), fx=0.5, fy=0.5)
fring = cv2.resize(cv2.imread("fring.jpg"), (0, 0), fx=0.5, fy=0.5)
walt = cv2.resize(cv2.imread("walt.jpg"), (0, 0), fx=0.5, fy=0.5)

cv2.imshow("Fring", walt)


while True:
	ret, frame = video.read()

	final = np.zeros_like(frame)
	final_face = deepfake(walt, frame, final)

	cv2.imshow("Deepfake", final_face)
	cv2.imshow("Frame", frame)



	if (cv2.waitKey(10) == ord("q")):
		break

video.release()
cv2.destroyAllWindows()



#every time i get frame, i want to run deepfake and put it on copy of frame then display copy of frame