import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, make_dataset_test
from PIL import Image
import torch
import json
import numpy as np
import os.path as osp
from PIL import ImageDraw


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.diction = {}

        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5

        # load data list from pairs file
        human_names = []
        cloth_names = []
        with open(os.path.join(opt.dataroot, opt.datapairs), 'r') as f:
            for line in f.readlines():
                h_name, c_name = line.strip().split()
                human_names.append(h_name)
                cloth_names.append(c_name)
        self.human_names = human_names
        self.cloth_names = cloth_names
        self.dataset_size = len(human_names)

        # input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5

        # input A test (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset_test(self.dir_A))

        # input B (real images)
        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths)
        self.build_index(self.B_paths)

        dir_E = '_edge'
        self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)
        self.E_paths = sorted(make_dataset(self.dir_E))
        self.ER_paths = make_dataset(self.dir_E)

        dir_M = '_mask'
        self.dir_M = os.path.join(opt.dataroot, opt.phase + dir_M)
        self.M_paths = sorted(make_dataset(self.dir_M))
        self.MR_paths = make_dataset(self.dir_M)

        dir_MC = '_colormask'
        self.dir_MC = os.path.join(opt.dataroot, opt.phase + dir_MC)
        self.MC_paths = sorted(make_dataset(self.dir_MC))
        self.MCR_paths = make_dataset(self.dir_MC)

        dir_C = '_color'
        self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.CR_paths = make_dataset(self.dir_C)
        # self.build_index(self.C_paths)

        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset_test(self.dir_A))

    def random_sample(self, item):
        name = item.split('/')[-1]
        name = name.split('-')[0]
        lst = self.diction[name]
        new_lst = []
        for dir in lst:
            if dir != item:
                new_lst.append(dir)
        return new_lst[np.random.randint(len(new_lst))]

    def build_index(self, dirs):
        for k, dir in enumerate(dirs):
            name = dir.split('/')[-1]
            name = name.split('-')[0]

            # print(name)
            for k, d in enumerate(dirs[max(k-20, 0):k+20]):
                if name in d:
                    if name not in self.diction.keys():
                        self.diction[name] = []
                        self.diction[name].append(d)
                    else:
                        self.diction[name].append(d)

    def __getitem__(self, index):
      # Get names from the pairs file
      c_name = self.cloth_names[index]
      h_name = self.human_names[index]

      # Define paths using os.path.join for better portability
      A_path = osp.join(self.dir_A, h_name.replace(".jpg", ".png"))
      B_path = osp.join(self.dir_B, h_name)
      C_path = osp.join(self.dir_C, c_name)
      E_path = osp.join(self.dir_E, c_name)

      # Initialize an empty dictionary to hold the input data
      input_dict = {}

      # Define parameters for transformations
      # We open each image and convert them as needed

      # Load label map (A)
      try:
          A = Image.open(A_path).convert('L')
          params = get_params(self.opt, A.size)
          if self.opt.label_nc == 0:
              transform_A = get_transform(self.opt, params)
              A_tensor = transform_A(A.convert('RGB'))
          else:
              transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
              A_tensor = transform_A(A) * 255.0
          input_dict['label'] = A_tensor
      except Exception as e:
          print(f"Error loading label map from {A_path}: {e}")
          input_dict['label'] = None

      # Load real image (B)
      try:
          B = Image.open(B_path).convert('RGB')
          transform_B = get_transform(self.opt, params)
          B_tensor = transform_B(B)
          input_dict['image'] = B_tensor
      except Exception as e:
          print(f"Error loading real image from {B_path}: {e}")
          input_dict['image'] = None

      # Load color image (C)
      try:
          C = Image.open(C_path).convert('RGB')
          C_tensor = transform_B(C)
          input_dict['color'] = C_tensor
      except Exception as e:
          print(f"Error loading color image from {C_path}: {e}")
          input_dict['color'] = None

      # Load edge image (E)
      try:
          E = Image.open(E_path).convert('L')
          E_tensor = transform_A(E)
          input_dict['edge'] = E_tensor
      except Exception as e:
          print(f"Error loading edge image from {E_path}: {e}")
          input_dict['edge'] = None

      # Load pose data
      pose_path = B_path.replace('.jpg', '_keypoints.json').replace(
          'test_img', 'test_pose')
      try:
          with open(pose_path, 'r') as f:
              pose_label = json.load(f)
              pose_data = pose_label['people'][0]['pose_keypoints']
              pose_data = np.array(pose_data).reshape((-1, 3))

          # Create pose map
          point_num = pose_data.shape[0]
          pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
          r = self.radius
          for i in range(point_num):
              pointx, pointy = pose_data[i, :2]
              if pointx > 1 and pointy > 1:
                  one_map = Image.new('L', (self.fine_width, self.fine_height))
                  draw = ImageDraw.Draw(one_map)
                  draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), fill='white')
                  one_map = transform_B(one_map.convert('RGB'))
                  pose_map[i] = one_map[0]

          input_dict['pose'] = pose_map
      except Exception as e:
          print(f"Error loading pose data from {pose_path}: {e}")
          input_dict['pose'] = None

      # Add path and name information
      input_dict['path'] = A_path
      input_dict['name'] = os.path.basename(A_path)

      # Return the input dictionary
      return input_dict


    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
