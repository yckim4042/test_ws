import numpy as np
from plyfile import PlyData, PlyElement

# 함수: 직사각형 경계와 원 배열을 가진 포인트 클라우드 생성
def create_point_cloud_with_circles_and_borders(board_width, board_height, circle_radius, rows, cols, circle_gap):
    point_cloud = []

    # 원 배열 생성 (원점 대칭)
    x_positions = np.linspace(-((cols - 1) * (circle_radius * 2 + circle_gap)) / 2, 
                              ((cols - 1) * (circle_radius * 2 + circle_gap)) / 2, cols)  # x 축 위치 (열 간 간격 9cm)
    y_positions = np.linspace(-((rows - 1) * (circle_radius * 2 + circle_gap)) / 2, 
                              ((rows - 1) * (circle_radius * 2 + circle_gap)) / 2, rows)  # y 축 위치 (행 간 간격 9cm)

    num_points_circle = 500  # 각 원을 표현할 포인트 개수
    theta = np.linspace(0, 2 * np.pi, num_points_circle)

    for x_center in x_positions:
        for y_center in y_positions:
            # 각 원의 좌표 생성
            x_circle = x_center + circle_radius * np.cos(theta)
            y_circle = y_center + circle_radius * np.sin(theta)
            z_circle = np.zeros(num_points_circle)
            points_circle = np.vstack((x_circle, y_circle, z_circle)).T
            point_cloud.append(points_circle)

            # 원점 대칭된 위치에 같은 원 생성
            x_circle_sym = -x_center + circle_radius * np.cos(theta)
            y_circle_sym = -y_center + circle_radius * np.sin(theta)
            points_circle_sym = np.vstack((x_circle_sym, y_circle_sym, z_circle)).T
            point_cloud.append(points_circle_sym)

    # 포인트 클라우드 병합
    point_cloud = np.concatenate(point_cloud, axis=0)
    return point_cloud

# PLY 파일로 포인트 클라우드 저장 함수
def save_point_cloud_to_ply(filename, point_cloud):
    vertices = [tuple(point) for point in point_cloud]
    vertex_data = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    ply_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([ply_element], text=True).write(filename)

# 보드 및 원 파라미터 설정
board_width = 0.5  # 50cm
board_height = 0.3  # 30cm
circle_radius = 0.03  # 3cm
rows = 3
cols = 4
circle_gap = 0.03  # 원들 중심 간의 간격 9cm

# 포인트 클라우드 생성 및 저장
point_cloud = create_point_cloud_with_circles_and_borders(board_width, board_height, circle_radius, rows, cols, circle_gap)
save_point_cloud_to_ply('./circle_borders.ply', point_cloud)

print("PLY 파일 생성 완료: rectangle_with_circle_borders.ply")

