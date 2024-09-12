import numpy as np
import open3d as o3d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 함수: PLY 파일에서 포인트 클라우드 로드
def load_point_cloud(filename):
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)
    return points

# 함수: RMSE 계산
def compute_rmse(boundary_points, target_points):
    distances = []
    for point in boundary_points:
        # 각 boundary의 점에 대해 target에서 가장 가까운 점 찾기
        diff = target_points - point
        dist = np.sqrt(np.sum(diff**2, axis=1))
        min_dist = np.min(dist)
        distances.append(min_dist)
    rmse = np.sqrt(np.mean(np.array(distances)**2))
    return rmse

# 함수: 회전 및 평행 이동 적용
def apply_transformation(points, rotation_matrix, translation_vector):
    return np.dot(points, rotation_matrix.T) + translation_vector

# 최적화 대상 함수: 회전 및 평행 이동 후 RMSE 계산
def optimization_function(params, boundary_points, target_points):
    theta = params[0]  # 회전각 (라디안)
    tx, ty = params[1], params[2]  # 평행 이동

    # 회전 행렬 생성
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    
    # 평행 이동 벡터 생성
    translation_vector = np.array([tx, ty])

    # 변환된 boundary 포인트들
    transformed_boundary = apply_transformation(boundary_points, rotation_matrix, translation_vector)

    # 변환된 포인트들에 대한 RMSE 계산
    return compute_rmse(transformed_boundary, target_points)

# PLY 파일 로드
target_points = load_point_cloud('circle_borders.ply')[:, :2]  # z 좌표 제거, 2D 포인트 사용
boundary_points = load_point_cloud('boundary.ply')[:, :2]

# 초기 값: 회전 각도 0, 평행 이동 없음
initial_params = [0, 0, 0]

# 최적화 실행 (RMSE가 최소화되는 회전 및 평행 이동 찾기)
result = minimize(optimization_function, initial_params, args=(boundary_points, target_points))

# 최적화된 회전각, 평행 이동 값
optimal_theta = result.x[0]
optimal_tx = result.x[1]
optimal_ty = result.x[2]

# 최적화된 변환 적용
optimal_rotation_matrix = np.array([[np.cos(optimal_theta), -np.sin(optimal_theta)],
                                    [np.sin(optimal_theta), np.cos(optimal_theta)]])
optimal_translation_vector = np.array([optimal_tx, optimal_ty])

aligned_boundary_points = apply_transformation(boundary_points, optimal_rotation_matrix, optimal_translation_vector)

# 최종 RMSE 계산
final_rmse = compute_rmse(aligned_boundary_points, target_points)
print(f"최적화된 RMSE: {final_rmse}")

# 시각화: 정렬된 포인트 클라우드와 원본 시각화
plt.figure(figsize=(8, 6))
plt.scatter(target_points[:, 0], target_points[:, 1], label='Target', color='blue', s=10)
plt.scatter(boundary_points[:, 0], boundary_points[:, 1], label='Boundary (Original)', color='red', s=10)
plt.scatter(aligned_boundary_points[:, 0], aligned_boundary_points[:, 1], label='Boundary (Aligned)', color='green', s=10, alpha=0.7)
plt.legend()
plt.title(f'Alignment of Boundary to Target\nFinal RMSE: {final_rmse:.4f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

