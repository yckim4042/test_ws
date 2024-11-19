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

# 함수: PCA를 적용하여 첫 번째 주성분(방향 벡터)을 계산
def pcaFindComponent(points):
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # 첫 번째 주성분을 반환
    return eigenvectors[:, np.argmax(eigenvalues)]

# 함수: 주성분 방향과 x축이 이루는 각도 계산
def calculate_initial_angle(first_principal_component):
    x_axis = np.array([1, 0])  # 2D에서의 x축
    cos_theta = np.dot(first_principal_component, x_axis) / np.linalg.norm(first_principal_component)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 각도 제한
    if first_principal_component[1] < 0:  # y축 방향을 고려하여 각도 부호 결정
        theta = -theta
    return theta

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
boundary_points = load_point_cloud('theirs_1_boundary.ply')[:, :2]

# PCA를 통해 첫 번째 주성분 계산
first_principal_component = pcaFindComponent(boundary_points)

# 첫 번째 주성분이 x축과 이루는 각도를 계산
initial_theta = calculate_initial_angle(first_principal_component)

# 초기 값: PCA에 의해 계산된 회전 각도와 평행 이동 없음
initial_params = [-initial_theta, 0, 0]

# 각도 제한 설정 (라디안으로 변환, 10도 = π/18)
theta_bounds = (-initial_theta-np.pi / 18, -initial_theta+np.pi / 18)  # -10도 ~ 10도
translation_bounds = (-np.inf, np.inf)  # 평행 이동에 제한 없음

# 경계 설정 (각도는 theta_bounds로 제한, 평행 이동은 제한 없음)
bounds = [theta_bounds, translation_bounds, translation_bounds]

# 최적화 실행 (RMSE가 최소화되는 회전 및 평행 이동 찾기, method='L-BFGS-B' 사용)
result = minimize(optimization_function, initial_params, args=(boundary_points, target_points), 
                  bounds=bounds, method='L-BFGS-B')

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
plt.scatter(target_points[:, 0], target_points[:, 1], label='Target', color='lime', s=3)
plt.scatter(aligned_boundary_points[:, 0], aligned_boundary_points[:, 1], label='Boundary (Aligned)', color='red', s=10, alpha=0.7)
plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1.12))  # 라벨을 더 오른쪽 위 끝으로 이동
plt.title(f'Alignment of Boundary to Target\nFinal RMSE: {final_rmse:.4f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig('theirs_1.png', dpi=300)

plt.show()

