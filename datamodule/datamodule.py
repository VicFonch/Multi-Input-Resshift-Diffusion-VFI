from imports.common_imports import *

class TargetImageDataset(Dataset):
    def __init__(self, data_dir, 
                 mean = [0.5, 0.5, 0.5], 
                 sd = [0.5, 0.5, 0.5],
                 **ignore_kwargs
        ):
        self.data_dir = data_dir

        subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

        self.image_paths = []
        for folder in subfolders:
            image_folder_path = os.path.join(data_dir, folder)
            image_paths = os.path.join(image_folder_path, f'frame{2}.jpg')
            self.image_paths.append(image_paths)

        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            #transforms.Normalize([0.8750041, 0.8435287, 0.8396906], [0.2153176, 0.2438267, 0.2413682])
            transforms.Normalize(mean, sd)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        images = self.transform_img(image)
        return images

class TwoImagesDataset(Dataset):
    def __init__(self, data_dir, 
                 mean = [0.5, 0.5, 0.5], 
                 sd = [0.5, 0.5, 0.5],
                 **ignore_kwargs
        ):
        self.data_dir = data_dir

        # Obtener la lista de carpetas en el directorio de datos
        subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

        # Crear una lista de rutas de acceso a las imágenes y flujos ópticos
        self.image_paths = []
        for folder in subfolders:
            image_folder_path = os.path.join(data_dir, folder)
            image_paths = [os.path.join(image_folder_path, f'frame{i}.jpg') for i in range(1, 3)]
            self.image_paths.append(image_paths)

        self.transform_img = Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            #transforms.Normalize([0.8750041, 0.8435287, 0.8396906], [0.2153176, 0.2438267, 0.2413682])
            transforms.Normalize(mean, sd)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_paths = self.image_paths[index]
        images = [Image.open(image_path) for image_path in image_paths]
        images = [self.transform_img(image) for image in images]
        images = torch.stack(images)
        return images

# class TripletImagesDataset(Dataset):
#     def __init__(self, data_dir, 
#                  mean = [0.5, 0.5, 0.5], 
#                  sd = [0.5, 0.5, 0.5],
#                  **ignore_kwargs
#         ):
#         self.data_dir = data_dir

#         # Obtener la lista de carpetas en el directorio de datos
#         subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

#         # Crear una lista de rutas de acceso a las imágenes y flujos ópticos
#         self.image_paths = []
#         for folder in subfolders:
#             image_folder_path = os.path.join(data_dir, folder)
#             image_paths = [os.path.join(image_folder_path, f'frame{i}.jpg') for i in range(1, 4)]
#             self.image_paths.append(image_paths)

#         self.transform_img = Compose([
#             transforms.ToTensor(),
#             transforms.Resize((256, 256)),
#             #transforms.Normalize([0.8750041, 0.8435287, 0.8396906], [0.2153176, 0.2438267, 0.2413682])
#             transforms.Normalize(mean, sd)
#         ])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         image_paths = self.image_paths[index]
#         images = [Image.open(image_path) for image_path in image_paths]
#         images = [self.transform_img(image) for image in images]
#         images = torch.stack(images)
#         return images

class TripletImagesDataset(Dataset):
    def __init__(self,
                 data_dir, 
                 mean = [0.5, 0.5, 0.5], 
                 sd = [0.5, 0.5, 0.5],
                 **ignore_kwargs):
        self.data_dir = data_dir
        self.mean = mean
        self.sd = sd

        # Obtener la lista de carpetas en el directorio de datos
        subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

        self.image_paths = []
        for folder in subfolders:
            image_folder_path = os.path.join(data_dir, folder)
            I0 = os.path.join(image_folder_path, 'frame1.jpg')
            It = os.path.join(image_folder_path, 'frame2.jpg')
            I1 = os.path.join(image_folder_path, 'frame3.jpg')
            self.image_paths.append([I0, It, I1])

        self.transform_img = Compose([
            transforms.ToTensor(),
            #transforms.Resize((544, 960)),
            #transforms.Resize((448, 896)),
            #transforms.Resize((384, 768)),
            transforms.Resize((256, 448)),
            #transforms.Normalize([0.8750041, 0.8435287, 0.8396906], [0.2153176, 0.2438267, 0.2413682])
            transforms.Normalize(self.mean, self.sd)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_paths = self.image_paths[index]
        images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        images = [self.transform_img(image) for image in images]
        return images

class TwoImagesWithFlowDataset(Dataset):
    def __init__(self, data_dir, 
                 mean = [0.5, 0.5, 0.5], 
                 sd = [0.5, 0.5, 0.5],
                 **ignore_kwargs
        ):
        self.data_dir = data_dir
        subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

        self.image_paths = []
        self.flow_paths = []
        for folder in subfolders:
            image_folder_path = os.path.join(data_dir, folder)
            flow_folder_path = os.path.join(data_dir, folder)
            
            image_paths = [os.path.join(image_folder_path, f'frame{i}.jpg') for i in range(1, 3)]
            flow_path = os.path.join(flow_folder_path, f'flow0tot.pt')
            
            self.image_paths.append(image_paths)
            self.flow_paths.append(flow_path)

        self.transform_img = Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean, sd)
        ])
        self.transform_flow = Resize((256, 256))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_paths = self.image_paths[index]
        flow_path = self.flow_paths[index]
        
        images = [Image.open(image_path) for image_path in image_paths]
        flow = torch.load(flow_path)
        
        images = [self.transform_img(image) for image in images]
        flow = self.transform_flow(flow.squeeze(0))

        images = torch.stack(images)

        return {'images': images, 'flow': flow}


class TripletImagesWithFlowsDataset(Dataset):
    def __init__(self, data_dir, 
                 mean = [0.5, 0.5, 0.5], 
                 sd = [0.5, 0.5, 0.5],
                 **ignore_kwargs
        ):
        self.data_dir = data_dir
        subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

        self.image_paths = []
        self.flow_paths = []
        for folder in subfolders:
            image_folder_path = os.path.join(data_dir, folder)
            flow_folder_path = os.path.join(data_dir, folder)
            
            image_paths = [os.path.join(image_folder_path, f'frame{i}.jpg') for i in range(1, 4)]
            flow_paths = [os.path.join(flow_folder_path, f'flow{i}.pt') for i in range(2)]
            
            self.image_paths.append(image_paths)
            self.flow_paths.append(flow_paths)

        self.transform_img = Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean, sd)
        ])
        self.transform_flow = Resize((256, 256))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_paths = self.image_paths[index]
        flow_paths = self.flow_paths[index]
        
        images = [Image.open(image_path) for image_path in image_paths]
        flows = [torch.load(flow_path) for flow_path in flow_paths]
        
        images = [self.transform_img(image) for image in images]
        flows = [self.transform_flow(flow.squeeze(0)) for flow in flows]

        images = torch.stack(images)
        flows = torch.stack(flows)

        return {'images': images, 'flows': flows}
    


