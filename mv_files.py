import os, shutil, h5py
from os.path import join as pjoin

root_dir = pjoin(os.getcwd(), "dataset/train")

# for i, file in enumerate(os.listdir(root_dir)):
#     pos = i % 5
#     p_dir = pjoin(os.getcwd(), "dataset/val_%d" % pos)
#     if not os.path.exists(p_dir): os.mkdir(p_dir)
#     fpath, fname = os.path.split(file)
#     dstfile = pjoin(root_dir, fname)
#     srcfile = pjoin(p_dir, fname)
#     shutil.copy(dstfile, srcfile)

# for i in range(5):
#     p = pjoin(os.getcwd(), "dataset/train_%d" % i)
#     for _, file in enumerate(os.listdir(p)):
#         fpath, fname = os.path.split(file)
#         dstfile = pjoin(p, fname)
#         srcfile = pjoin(root_dir, fname)
#         shutil.copy(dstfile, srcfile)

sum = 0
for i in range(5):
    p = pjoin(os.getcwd(), "dataset/train_%d" % i)
    sum += len(os.listdir(p))
    print(i, len(os.listdir(p)))
# print(sum)

for i in range(5):
    p = pjoin(os.getcwd(), "dataset/train_%d" % i)
    sum += len(os.listdir(p))
    print(i, len(os.listdir(p)))
# print(len(os.listdir(root_dir)))
#     
    # p_dir = pjoin(os.getcwd(), "dataset/interm_data/test_intermediate_%d" % pos)
    # 
    # p = pjoin(p_dir,"raw")
    # if not os.path.exists(p): os.mkdir(p)
    # sum += len(os.listdir(p))
    # print(i, len(os.listdir(p)))
    # fpath, fname = os.path.split(file)
    # dstfile = pjoin(input_path, fname)
    # srcfile = pjoin(root_dir, "train_intermediate_%d/raw" % (j), fname)
    # shutil.move(dstfile, srcfile)

# for sub_dir in ['train', 'val', 'test_obs']:
    # p = pjoin(os.getcwd(), "dataset/raw_data", sub_dir)
    # print(f"len of %s is %d" % (p, len(os.listdir(p))))


# input_path = pjoin(root_dir, "val_intermediate_8/raw")
# # print(input_path)

# for j in range(5):
#     input_path = pjoin(root_dir, "train_intermediate_%d/raw" % (j + 5))
#     print(input_path)
#     n = len(os.listdir(input_path))
#     for i, file in enumerate(os.listdir(input_path)):
#         fpath, fname = os.path.split(file)
#         dstfile = pjoin(input_path, fname)
#         srcfile = pjoin(root_dir, "train_intermediate_%d/raw" % (j), fname)
#         shutil.move(dstfile, srcfile)

# for i, file in enumerate(os.listdir(input_path)[:26]):
#     fpath, fname = os.path.split(file)
#     dstfile = pjoin(input_path, fname)
#     srcfile = pjoin(root_dir, "val_intermediate_%d/raw    " % 9, fname)
#     shutil.move(dstfile, srcfile)

# for j in range(3):
#     input_path = pjoin(root_dir, "test_intermediate_%d/raw" % (j))
#     print(input_path)
#     n = len(os.listdir(input_path))
#     for i, file in enumerate(os.listdir(input_path)[:(n//2)]):
#         fpath, fname = os.path.split(file)
#         dstfile = pjoin(input_path, fname)
#         srcfile = pjoin(root_dir, "test_intermediate_%d/raw" % (j + 5), fname)
#         shutil.move(dstfile, srcfile)




# print("val: ")
# for i in range(10):
#     p_dir = pjoin(root_dir, "val_intermediate_%d" % i)
#     if not os.path.exists(p_dir): os.mkdir(p_dir)
#     p = pjoin(p_dir,"raw")
#     if not os.path.exists(p): os.mkdir(p)
#     sum += len(os.listdir(p))
#     print(i, len(os.listdir(p)))
# print(sum)
# print("test: ")
# for i in range(3):
#     p_dir = pjoin(os.getcwd(), "dataset/interm_data/test_intermediate_%d" % i)
#     if not os.path.exists(p_dir): os.mkdir(p_dir)
#     p = pjoin(p_dir,"raw")
#     if not os.path.exists(p): os.mkdir(p)
#     sum += len(os.listdir(p))
#     print(p, len(os.listdir(p)))
# print(sum)

# # print(len(os.listdir(pjoin(root_dir, "val_intermediate_%d/raw"))))
# for i in range(4):
#     print(len(os.listdir(pjoin(root_dir, "val_intermediate_%d/raw" % i))))

# print(len(os.listdir(pjoin(root_dir, "test_intermediate/raw"))))
# print(len(os.listdir(pjoin(root_dir, "val_intermediate/raw"))))

# input_path = pjoin(root_dir, "val_intermediate_2/raw")
# print(input_path)

# for i, file in enumerate(os.listdir(input_path)):
#     # pos = i // 20000
#     fpath, fname = os.path.split(file)
#     dstfile = pjoin(input_path, fname)
#     srcfile = pjoin(root_dir, "val_intermediate_%d/raw" % 3, fname)
#     shutil.move(dstfile, srcfile)
#     if i == 1: break


# for i in range(4):
#     print(len(os.listdir(pjoin(root_dir, "train_intermediate_%d/raw" % i))))

# print()

# for i in range(4):
#     print(len(os.listdir(pjoin(root_dir, "val_intermediate_%d/raw" % i))))