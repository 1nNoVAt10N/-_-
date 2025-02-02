import os
import random
import sklearn
# 假设所有图片路径存储在这个列表中
cache_dir = "./cache_384"  # 替换为你的图片缓存路径
all_images = [os.path.join(cache_dir, img) for img in os.listdir(cache_dir) if img.endswith('.npz')]
all_images = [img.replace('\\', '/') for img in all_images]
total_images = len(all_images)
assert total_images >= 10, "图片数量太少，无法进行合理划分。"
random.seed(42) 
random.shuffle(all_images)

train_size = int(0.8 * total_images)

test_size = total_images - train_size 
train_images = all_images[:train_size]

test_images = all_images[train_size :]

print(f"总图片数量: {total_images}")
print(f"训练集: {len(train_images)} 张")

print(f"测试集: {len(test_images)} 张")

with open("train_images.txt", "w") as f:
    f.writelines(f"{path}\n" for path in train_images)



with open("test_images.txt", "w") as f:
    f.writelines(f"{path}\n" for path in test_images)


