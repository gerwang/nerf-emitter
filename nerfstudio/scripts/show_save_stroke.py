import argparse
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

# 全局变量
drawing = False  # 当鼠标按下时变为True
path_coordinates = []  # 存储轨迹坐标
images = []

parser = argparse.ArgumentParser(description='''''')
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()
# 创建空白画布
original_image = cv2.imread(args.input)

image_all_stroke = original_image.copy()


def clear_coordinates():
    global image_all_stroke
    path_coordinates.clear()
    images.clear()
    image_all_stroke = original_image.copy()


def add_coordinate(x, y):
    draw_circle(image_all_stroke, x, y)
    image_copy = original_image.copy()
    draw_circle(image_copy, x, y)
    images.append(image_copy)
    path_coordinates.append((x, y))


def draw_circle(img, x, y):
    # 圆的参数
    center_coordinates = (x, y)  # 圆心，图像中心
    radius = 5  # 圆的半径
    color = (64, 255, 64)  # 白色填充
    thickness = -1  # 表示圆将是填充的

    # 画填充的圆
    cv2.circle(img, center_coordinates, radius, color, thickness, lineType=cv2.LINE_AA)


# 回调函数，用于处理鼠标事件
def draw_path(event, x, y, flags, param):
    global drawing, path_coordinates, images

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        clear_coordinates()
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if len(path_coordinates) == 0 or path_coordinates[-1][0] != x or path_coordinates[-1][1] != y:
                add_coordinate(x, y)


# 设置窗口和回调函数
cv2.namedWindow('DrawPath')
cv2.setMouseCallback('DrawPath', draw_path)

while True:
    cv2.imshow('DrawPath', image_all_stroke)

    # 按下 's' 键保存轨迹坐标到文件
    if cv2.waitKey(1) & 0xFF == ord('s') and path_coordinates:
        os.makedirs(args.output, exist_ok=True)
        with open(f'{args.output}/path_coordinates.txt', 'w') as file:
            for point in path_coordinates:
                file.write(f'{point[0]} {point[1]}\n')
        os.makedirs(f'{args.output}/images', exist_ok=True)
        for i, image in enumerate(images):
            cv2.imwrite(f'{args.output}/images/{i}.png', image)
        clear_coordinates()

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
