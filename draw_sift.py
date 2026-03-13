# import cv2

# # 读取两张图
# img1 = cv2.imread("experient_fig/sift/sift1.png")
# img2 = cv2.imread("experient_fig/sift/sift2.png")

# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # 创建SIFT
# sift = cv2.SIFT_create()

# # 检测关键点并计算描述子
# kp1, des1 = sift.detectAndCompute(gray1, None)
# kp2, des2 = sift.detectAndCompute(gray2, None)

# # BF匹配
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# # Lowe ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append(m)

# # 画匹配结果
# img_match = cv2.drawMatches(img1, kp1, img2, kp2, good, None)

# cv2.imwrite("experient_fig/sift/siftResult/sift_match.jpg", img_match)























# import cv2
# import os

# # ===============================
# # 配置参数
# # ===============================
# img1_path = "experient_fig/sift/sift1.png"
# img2_path = "experient_fig/sift/sift2.png"
# save_dir = "experient_fig/sift/siftResult1"
# save_name = "sift_match.jpg"
# top_N = 100  # 只画前 top_N 个匹配

# # ===============================
# # 创建保存文件夹
# # ===============================
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, save_name)

# # ===============================
# # 读取图像
# # ===============================
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)

# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # ===============================
# # SIFT检测关键点 & 描述子
# # ===============================
# sift = cv2.SIFT_create()
# kp1, des1 = sift.detectAndCompute(gray1, None)
# kp2, des2 = sift.detectAndCompute(gray2, None)

# # ===============================
# # BF匹配 + Lowe ratio test
# # ===============================
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# good = []
# for m, n in matches:
#     if m.distance < 0.8 * n.distance:
#         good.append(m)

# # ===============================
# # 按距离排序 & 只保留前 top_N 个匹配
# # ===============================
# good = sorted(good, key=lambda x: x.distance)
# good = good[:top_N]

# # ===============================
# # 可选：RANSAC 筛选内点
# # ===============================
# pts1 = cv2.KeyPoint_convert([kp1[m.queryIdx] for m in good])
# pts2 = cv2.KeyPoint_convert([kp2[m.trainIdx] for m in good])
# pts1 = pts1.reshape(-1,1,2)
# pts2 = pts2.reshape(-1,1,2)
# H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 10.0)
# good = [m for m, keep in zip(good, mask.ravel()) if keep]  # 只保留内点

# # ===============================
# # 画匹配结果
# # ===============================
# img_match = cv2.drawMatches(
#     img1, kp1, img2, kp2, good, None,
#     matchColor=(0,255,0),  # 绿色线条
#     singlePointColor=(255,0,0),  # 蓝色关键点
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
# )

# # ===============================
# # 保存结果
# # ===============================
# cv2.imwrite(save_path, img_match)
# print("SIFT匹配结果已保存到:", save_path)
















# import cv2
# import numpy as np
# import os

# # -----------------------------
# # 路径设置
# # -----------------------------
# img1_path = "experient_fig/sift/sift3.png"
# img2_path = "experient_fig/sift/sift4.png"
# save_dir = "experient_fig/sift/siftResult"

# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, "orb_knn_ransac_matches.png")

# # -----------------------------
# # 读取图片
# # -----------------------------
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)

# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # -----------------------------
# # ORB特征提取
# # -----------------------------
# orb = cv2.ORB_create(nfeatures=2000)

# kp1, des1 = orb.detectAndCompute(gray1, None)
# kp2, des2 = orb.detectAndCompute(gray2, None)

# # -----------------------------
# # KNN匹配
# # -----------------------------
# bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# knn_matches = bf.knnMatch(des1, des2, k=2)

# # -----------------------------
# # Lowe's Ratio Test
# # -----------------------------
# good_matches = []

# for m, n in knn_matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append(m)

# # -----------------------------
# # 提取匹配点
# # -----------------------------
# pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
# pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# # -----------------------------
# # RANSAC计算单应矩阵
# # -----------------------------
# H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 10.0)

# mask = mask.ravel().tolist()

# # -----------------------------
# # 分离inlier和outlier
# # -----------------------------
# inlier_matches = []
# outlier_matches = []

# for i, m in enumerate(good_matches):
#     if mask[i]:
#         inlier_matches.append(m)
#     else:
#         outlier_matches.append(m)

# # -----------------------------
# # 只取前20个inlier
# # -----------------------------
# inlier_matches = sorted(inlier_matches, key=lambda x: x.distance)
# top_inliers = inlier_matches[:20]

# # -----------------------------
# # 画图
# # -----------------------------

# # 先画outlier（灰色淡化）
# result = cv2.drawMatches(
#     img1, kp1,
#     img2, kp2,
#     outlier_matches,
#     None,
#     matchColor=(200,200,200),
#     singlePointColor=(200,200,200),
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
# )

# # 再叠加inlier（高亮绿色）
# result = cv2.drawMatches(
#     img1, kp1,
#     img2, kp2,
#     top_inliers,
#     result,
#     matchColor=(0,255,0),
#     singlePointColor=(0,255,0),
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
# )

# # -----------------------------
# # 保存结果
# # -----------------------------
# cv2.imwrite(save_path, result)

# print("匹配结果已保存:", save_path)



# import cv2
# import numpy as np
# import os

# # -----------------------------
# # 路径设置
# # -----------------------------
# img1_path = "experient_fig/sift/sift3.png"
# img2_path = "experient_fig/sift/sift4.png"
# save_dir = "experient_fig/sift/siftResult"

# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, "sift_knn_ransac_matches.png")

# # -----------------------------
# # 读取图像
# # -----------------------------
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)

# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # -----------------------------
# # SIFT特征提取
# # -----------------------------
# sift = cv2.SIFT_create(nfeatures=2000)

# kp1, des1 = sift.detectAndCompute(gray1, None)
# kp2, des2 = sift.detectAndCompute(gray2, None)

# # -----------------------------
# # FLANN匹配器 (SIFT推荐)
# # -----------------------------
# index_params = dict(algorithm=1, trees=5)  # KDTree
# search_params = dict(checks=50)

# flann = cv2.FlannBasedMatcher(index_params, search_params)

# knn_matches = flann.knnMatch(des1, des2, k=2)

# # -----------------------------
# # Lowe Ratio Test
# # -----------------------------
# good_matches = []

# for m, n in knn_matches:
#     if m.distance < 0.8 * n.distance:
#         good_matches.append(m)

# # -----------------------------
# # 提取匹配点
# # -----------------------------
# pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
# pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# # -----------------------------
# # RANSAC
# # -----------------------------
# H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

# mask = mask.ravel().tolist()

# inliers = []
# outliers = []

# for i, m in enumerate(good_matches):
#     if mask[i]:
#         inliers.append(m)
#     else:
#         outliers.append(m)

# # 只取前20个inlier
# inliers = sorted(inliers, key=lambda x: x.distance)
# top_inliers = inliers[:20]

# # -----------------------------
# # 绘图
# # -----------------------------
# result = cv2.drawMatches(
#     img1, kp1,
#     img2, kp2,
#     outliers,
#     None,
#     matchColor=(200,200,200),
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
# )

# result = cv2.drawMatches(
#     img1, kp1,
#     img2, kp2,
#     top_inliers,
#     result,
#     matchColor=(0,255,0),
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
# )

# # -----------------------------
# # 保存
# # -----------------------------
# cv2.imwrite(save_path, result)

# print("SIFT匹配完成")
# print("kp1:", len(kp1))
# print("kp2:", len(kp2))
# print("good matches:", len(good_matches))
# print("RANSAC inliers:", len(inliers))
# print("保存路径:", save_path)







# import cv2
# import numpy as np
# import os

# # -----------------------------
# # 路径
# # -----------------------------
# img1_path = "experient_fig/sift/nearsift/Snipaste_2026-03-10_17-10-21.png"
# img2_path = "experient_fig/sift/nearsift/Snipaste_2026-03-10_17-11-18.png"
# save_dir = "experient_fig/sift/siftResult"

# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, "akaze_knn_ransac_matches.png")

# # -----------------------------
# # 读取图片
# # -----------------------------
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)

# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # -----------------------------
# # AKAZE特征提取
# # -----------------------------
# akaze = cv2.AKAZE_create()

# kp1, des1 = akaze.detectAndCompute(gray1, None)
# kp2, des2 = akaze.detectAndCompute(gray2, None)

# # -----------------------------
# # BFMatcher
# # -----------------------------
# bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# knn_matches = bf.knnMatch(des1, des2, k=2)

# # -----------------------------
# # Ratio Test
# # -----------------------------
# good_matches = []

# for m, n in knn_matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append(m)

# # -----------------------------
# # 提取匹配点
# # -----------------------------
# pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
# pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# # -----------------------------
# # RANSAC
# # -----------------------------
# H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 10.0)

# mask = mask.ravel().tolist()

# inliers = []
# outliers = []

# for i, m in enumerate(good_matches):
#     if mask[i]:
#         inliers.append(m)
#     else:
#         outliers.append(m)

# # 只取前20个inlier
# inliers = sorted(inliers, key=lambda x: x.distance)
# top_inliers = inliers[:20]

# # -----------------------------
# # 绘制匹配
# # -----------------------------
# result = cv2.drawMatches(
#     img1, kp1,
#     img2, kp2,
#     outliers,
#     None,
#     matchColor=(200,200,200),
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
# )

# result = cv2.drawMatches(
#     img1, kp1,
#     img2, kp2,
#     top_inliers,
#     result,
#     matchColor=(0,255,0),
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
# )

# # -----------------------------
# # 保存
# # -----------------------------
# cv2.imwrite(save_path, result)

# print("AKAZE匹配完成")
# print("kp1:", len(kp1))
# print("kp2:", len(kp2))
# print("good matches:", len(good_matches))
# print("RANSAC inliers:", len(inliers))
# print("保存路径:", save_path)
# np.set_printoptions(suppress=True, precision=6)
# print("Homography matrix:")
# print(H)




import cv2
import numpy as np
import os

# -----------------------------
# 路径
# -----------------------------
img1_path = "experient_fig/sift/000505.jpg"
img2_path = "experient_fig/sift/000517.jpg"
save_dir = "experient_fig/sift/siftResult"

os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "akaze_knn_ransac_matches.png")
warp_path = os.path.join(save_dir, "img1_warp_to_img2.png")
rematch_path = os.path.join(save_dir, "akaze_rematch_after_warp.png")

# -----------------------------
# 读取图片
# -----------------------------
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# -----------------------------
# AKAZE特征提取
# -----------------------------
akaze = cv2.AKAZE_create()

kp1, des1 = akaze.detectAndCompute(gray1, None)
kp2, des2 = akaze.detectAndCompute(gray2, None)

# -----------------------------
# BFMatcher
# -----------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
knn_matches = bf.knnMatch(des1, des2, k=2)

# -----------------------------
# Ratio Test
# -----------------------------
good_matches = []
for m, n in knn_matches:
    if m.distance < 0.65 * n.distance:
        good_matches.append(m)

# -----------------------------
# 提取匹配点
# -----------------------------
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# -----------------------------
# RANSAC求单应矩阵
# -----------------------------
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 10.0)

mask = mask.ravel().tolist()

inliers = []
outliers = []

for i, m in enumerate(good_matches):
    if mask[i]:
        inliers.append(m)
    else:
        outliers.append(m)

# 只取前20个inlier
inliers = sorted(inliers, key=lambda x: x.distance)
top_inliers = inliers[:20]

# -----------------------------
# 绘制第一次匹配结果
# -----------------------------
result = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    outliers,
    None,
    matchColor=(200, 200, 200),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

result = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    top_inliers,
    result,
    matchColor=(0, 255, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(save_path, result)

# -----------------------------
# 透视变换：把img1映射到img2视角
# -----------------------------
warped1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
cv2.imwrite(warp_path, warped1)

# -----------------------------
# 在逆透视/配准后的图上重新做AKAZE
# -----------------------------
warped_gray1 = cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY)

kpw, desw = akaze.detectAndCompute(warped_gray1, None)
kp2_new, des2_new = akaze.detectAndCompute(gray2, None)

knn_matches2 = bf.knnMatch(desw, des2_new, k=2)

good_matches2 = []
for m, n in knn_matches2:
    if m.distance < 0.8 * n.distance:
        good_matches2.append(m)

# 取前30个最佳匹配
good_matches2 = sorted(good_matches2, key=lambda x: x.distance)
top_matches2 = good_matches2[:30]

rematch_result = cv2.drawMatches(
    warped1, kpw,
    img2, kp2_new,
    top_matches2,
    None,
    matchColor=(0, 255, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(rematch_path, rematch_result)

# -----------------------------
# 输出信息
# -----------------------------
print("AKAZE第一次匹配完成")
print("kp1:", len(kp1))
print("kp2:", len(kp2))
print("good matches:", len(good_matches))
print("RANSAC inliers:", len(inliers))

print("逆透视/配准图保存路径:", warp_path)
print("二次匹配结果保存路径:", rematch_path)

np.set_printoptions(suppress=True, precision=6)
print("Homography matrix:")
print(H)