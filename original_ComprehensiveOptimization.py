"""
Date&Time           2023/2/9 13:41
Author              YuWang

"""
# 0. 导入需要的库
import os
import random
import numpy as np
from sklearn import model_selection
import torchvision.transforms as transforms
from torch.utils import data as torch_data
from PIL import Image
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import torch
from torch.utils import data as torch_data
import datetime
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import tensorboard

# <1.1 读取数据路径与制作标签>
IMAGE_DATASET_PATH = "datasets"
class_name_s = []                 # 存储标签名
data_set = []                     # 存储图片地址和标签
for parent_class_name in os.listdir(IMAGE_DATASET_PATH):
    # 返回(路径)下所有文件夹名, ['Decks', 'Pavements', 'Walls']
    for sub_class_name in os.listdir(os.path.join(IMAGE_DATASET_PATH, parent_class_name)):
        # 拼接路径，例如：datasets\Decks返回路径下所有文件夹名
        class_name = ".".join([parent_class_name, sub_class_name])
        # 返回parent_class_name.sub_class_name作为标签名

        class_i = len(class_name_s)       # 返回class_name_s列表长度
        class_name_s.append(class_name)  # 将标签名存入class_name_s
        data_x = []                        # 存储图片路径
        data_y = []                        # 存储图片标签
        for sub_data_file_name in os.listdir(os.path.join(IMAGE_DATASET_PATH, parent_class_name, sub_class_name)):
            data_x.append(os.path.join(IMAGE_DATASET_PATH, parent_class_name, sub_class_name, sub_data_file_name))       # 遍历图片路径并存储在data_x
            data_y.append(class_i)          # 将标签存储在data_y
        data_set.append((data_x, data_y))  # 整合图片路径列表和标签列表

# <1.2.1 数据集划分>
x_train_s = []                    # 用来存储训练集图片路径
x_test_s = []                     # 用来存储测试集图片路径
y_train_s = []                    # 用来存储训练集标签
y_test_s = []                     # 用来存储测试集标签
for x_data, y_data in data_set:
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.2)           # 每个类别的所有图片路径和标签按8：2分为训练集和测试集，训练标签和测试标签
    x_train_s.extend(x_train)    # 六个类别的训练集路径依次存入x_train_s
    x_test_s.extend(x_test)      # 六个类别的测试集路径依次存入x_test_s
    y_train_s.extend(y_train)    # 六个类别的训练标签依次存入y_train_s
    y_test_s.extend(y_test)      # 六个类别的测试标签依次存入y_test_s
# 可以发现，上述划分训练集和测试集时并不是对整个数据按照比例进行划分，而是对每个类别分别进行划分后合并，这样做可以提高每个类别训练样本的均匀性。print(len(x_train_s)) = 44871, 训练集的数量约为总数的8/10。
train_data = list(zip(x_train_s, y_train_s))
# 将训练集和训练标签合并—>[('图片地址1', 标签1),……('图片地址44871', 标签44871)]
random.shuffle(train_data)
# 将列表中元素顺序打乱
x_train_s, y_train_s = list(zip(*train_data))
# 将打乱后的训练数据拆分成新的训练集和训练标签
test_data = list(zip(x_test_s, y_test_s))     # 对测试集路径做相同的打乱和拆分处理
random.shuffle(test_data)
x_test_s, y_test_s = list(zip(*test_data))
dataset_sizes = {
    "train": len(train_data),
    "val": len(test_data),
}                                                # 保存训练集和测试集数据量
# 计算样本权重，可以得到每个样本所占比重，是后续设置损失函数时所需参数
y_train_np = np.asarray(y_train_s)            # 将训练集标签转为数组
y_one_hot = np.eye(len(class_name_s))[y_train_np]   # 将训练集标签数组转为独热编码
class_count = np.sum(y_one_hot, axis=0)      # 列向量求和，可知道训练集中各类别的数量
total_count = np.sum(class_count)                     # 求六个类别总数量—>44871
label_weight = (1 / class_count) * total_count / 2  # 求样本权重

# <1.2.2图片预处理>
data_transforms ={
    'train': transforms.Compose([
        transforms.Resize((224, 224)),                    # 调整图片尺寸—>[224, 224]
        transforms.RandomGrayscale(p=0.1),                # 随机图片灰度化
        transforms.RandomAffine(0, shear=5, scale=(0.8, 1.2)), # 随机仿射变化
        transforms.ColorJitter(brightness=(
            0.5, 1.5), contrast=(0.8, 1.5), saturation=0),      # 图片属性变换
        transforms.RandomHorizontalFlip(),                # 随机图片水平翻转
        transforms.ToTensor(),                             # 将图片格式转换为张量
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225]), # 图片归一化
    ]),
    # 测试集图片预处理，只进行图片尺寸、格式和归一化处理
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225]),
    ]),
}

# <1.2.3图片信息读取>
class CrackDataset(torch_data.Dataset):    # 定义一个读取图片信息的类
    _files = None
    _labels = None
    _transform = False
    # 读取外部信息函数
    def __init__(self, abs_file_path_s, y_datas, trans=False):
        self._files = abs_file_path_s    # 传入图像地址
        self._labels = y_datas            # 传入图像标签
        self._transform = trans           # 传入需要载入的预处理函数
    # 图像处理函数，按顺序读取图片并将图片转换成RGB格式
    def __getitem__(self, item):
        img = Image.open(self._files[item]).convert("RGB")
        label = self._labels[item]        # 按顺序读取图片标签
        if self._transform:
            img = self._transform(img)    # 对图片进行预处理
        return img, label                  # 返回预处理后的图像信息和标签

    def __len__(self):
        return len(self._files)           # 返回数据长度

train_data = CrackDataset(abs_file_path_s=x_train_s, y_datas=y_train_s, trans=data_transforms["train"])          # 调用类功能，输出训练数据
test_data = CrackDataset(abs_file_path_s=x_test_s, y_datas=y_test_s, trans=data_transforms["val"])            # 调用类功能，输出测试数据

# <1.3 创建数据加载器dataloaders>
BATCH_SIZE = 256            # 一次输入网络模型的图片量
train_data_loader = torch_data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = torch_data.DataLoader(test_data, batch_size=BATCH_SIZE)

dataloaders = {
    'train': train_data_loader,
    'val': test_data_loader,
}                           # 将训练数据和测试数据存入字典dataloaders

# <2.1.1 创建框架中模型>
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
model = models.resnet18(pretrained=True)# 使用预训练模型
fc_inputs = model.fc.in_features
# 载入的resnet18不包含后续的全连接层，需要根据自己项目需求写入
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),                   #——以p=0.5添加项Dropout——
    nn.Linear(fc_inputs, 256),          # 全连接层，接256个神经元
    nn.ReLU(),                            # 激活
    nn.Dropout(p=0.5),                   #——以p=0.5添加项Dropout——
    nn.Linear(256, 6)  # 全连接层，接输出
)
model = model.to(device)                # 将由CPU保存的模型加载到GPU上，提高训练速度

# # <CommonBloc结构>
# class CommonBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, stride):  # 定义功能函数
#         super(CommonBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3,
#                                stride=stride, padding=1, bias=False)                      # 第一次卷积操作参数设置
#         self.bn1 = nn.BatchNorm2d(out_channel)
#         self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,
#                                stride=stride, padding=1, bias=False)                      # 第二次卷积操作参数设置
#         self.bn2 = nn.BatchNorm2d(out_channel)
#
#     def forward(self, x):                 # 调用上述功能函数，对输入x进行处理
#         identity = x                       # 将初始输入x直接赋给identity
#         x = F.relu(self.bn1(self.conv1(x)), inplace=True) # 对输入x进行第一次卷积操作并激活
#         x = self.bn2(self.conv2(x))       # 第二次卷积操作
#         x += identity                      # 将第二次卷积操作的输出与未经处理的输入相加
#         return F.relu(x, inplace=True)   # 激活后返回输出结果
# # <SpecialBlock结构>
# class SpecialBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, stride):  # 定义功能函数
#         super(SpecialBlock, self).__init__()
#         self.change_channel = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
#             nn.BatchNorm2d(out_channel)
#         )                                              # 旁支卷积，负责改变输入x维度
#         self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)           # 第一次卷积操作参数设置
#         self.bn1 = nn.BatchNorm2d(out_channel)
#         self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)           # 第二次卷积操作参数设置
#         self.bn2 = nn.BatchNorm2d(out_channel)
#
#     def forward(self, x):                  # 调用上述功能函数，对输入x进行处理
#         identity = self.change_channel(x) # 输入x经旁支卷积处理后赋给identity
#         x = F.relu(self.bn1(self.conv1(x)), inplace=True) # 对输入x进行第一次卷积操作并激活
#         x = self.bn2(self.conv2(x))        # 第二次卷积操作
#         x += identity   # 将第二次卷积操作的输出x与经旁支卷积处理后的identity相加
#         return F.relu(x, inplace=True)     # 激活后返回输出结果
# # <2.1.2 编译的模型>
# class ResNet18(nn.Module):
#     def __init__(self, classes_num = 6):   # 初始化分类数为6
#         super(ResNet18, self).__init__()
#         self.prepare = nn.Sequential(
#             nn.Conv2d(3, 64, 7, 2, 3),     # 预卷积操作参数设置
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),         # 预卷积操作后—> [batch, 64, 112, 112]
#             nn.MaxPool2d(3, 2, 1)          # 最大池化参数设置
#         )                                    # 池化后 —> [batch, 64, 56, 56]
#         self.layer1 = nn.Sequential(
#             CommonBlock(64, 64, 1),     # 第一个残差单元，—> [batch, 64, 56, 56]
#             CommonBlock(64, 64, 1)      # 第二个残差单元，—> [batch, 64, 56, 56]
#         )
#         self.layer2 = nn.Sequential(
#             SpecialBlock(64, 128, [2, 1]), # 第三个残差单元，—> [batch, 128, 28, 28]
#             CommonBlock(128, 128, 1)       # 第四个残差单元，—> [batch, 128, 28, 28]
#         )
#         self.layer3 = nn.Sequential(
#             SpecialBlock(128, 256, [2, 1]), # 第五个残差单元，—> [batch, 256, 14, 14]
#             CommonBlock(256, 256, 1)        # 第六个残差单元，—> [batch, 256, 14, 14]
#         )
#         self.layer4 = nn.Sequential(
#             SpecialBlock(256, 512, [2, 1]), # 第七个残差单元，—> [batch, 512, 7, 7]
#             CommonBlock(512, 512, 1)         # 第八个残差单元，—> [batch, 512, 7, 7]
#         )
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         # 通过一个自适应均值池化—> [batch, 512, 1, 1]
#         self.fc = nn.Sequential(
#             nn.Linear(512, 256),             # 全连接层，512—>256
#             nn.ReLU(inplace=True),
#             nn.Linear(256, classes_num)     # 六分类，256—> classes_num == 6
#         )
#
#     # 使用ResNet18对输入x进行处理，输入x—> [batch, 3, 224, 224]
#     def forward(self, x):
#         x = self.prepare(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.pool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
#         return x                             # 返回网络输出结果—>[batch, 6]
# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
# model = ResNet18()
# model = model.to(device)

# <EarlyStopping>
class EarlyStopping:       # 如果在给定的耐心值之后验证损失没有改善，则停止训练
    def __init__(self, patience=7, verbose=False, delta=0):
        """
            patience (int):上次验证集损失值改善后等待几个epoch
            verbose (bool):如果是True，为每个验证集损失值改善打印一条信息
            delta (float):监测数量的最小变化，以符合改进的要求
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        # 把每次的验证损失依次赋给score（取负值）
        # 这里需要注意，损失越小越好，这里取负，则越大越好，比较时如果大就更新best_score
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            # 当新score比best_score小，则继续训练，直至patience次数停止训练
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:     # 如果在patience次数内的某次score大，则更新best_score，重新计数
            self.best_score = score
            self.counter = 0

# <2.2 定义训练函数>
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    :param model:  模型
    :param criterion:  损失函数
    :param optimizer:  优化函数
    :param scheduler:  调整学习率
    :param num_epochs:  数据集训练组数
    """
    if not os.path.exists("model"):
        os.mkdir("model")
    MODEL_SAVE_PATH = os.path.join("model", "best.pt")

    writer = tensorboard.SummaryWriter(os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    # 创建"logs"文件夹，并以"训练开始日期-时间"为子文件名存储训练数据
    early_stopping = EarlyStopping(30)      # 插入patience设置为30
    best_acc = 0.0                          # 初始化最优准确率
    for epoch in range(num_epochs):
        since = datetime.datetime.now()   # 记录开始时间
        loss_both = {
        }                                    # 存储损失值
        acc_both = {
        }                                    # 存储准确率
        # 每一个epoch都包含训练集和测试集
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0              # 初始化损失值
            running_corrects = 0            # 初始化准确率

            # 开始循环训练，每次从dataloaders读取bach_size个图片和标签。
            for loop_i, datas in enumerate(dataloaders[phase]):
                inputs, labels = datas
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()       # 初始化优化梯度
                # 训练模式进行如下操作
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # 最后输出的6个结果为六个类别的概率，取最大概率的位置索引赋给preds
                    loss = criterion(outputs, labels)      # 计算输出与标签的损失
                    print(f"{phase}:{loop_i},loss:{loss}") # 打印每个bach_size损失值
                    # 训练模式下需要进行反向传播和参数优化
                    if phase == 'train':
                        loss.backward()        # 训练模式下计算损失
                        optimizer.step()       # 训练模式下参数优化方法
                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]  # 计算一个epoch损失值
            epoch_acc = running_corrects.double() / dataset_sizes[phase] # 计算一个epoch准确率

            loss_both[phase] = epoch_loss    # 将每个epoch损失值存入字典
            acc_both[phase] = epoch_acc      # 将每个epoch准确率存入字典
        scheduler.step()                      # 调整学习率

        time_elapsed = datetime.datetime.now() - since         # 计算一个epoch时间
        print(
            f"time :{time_elapsed}, epoch :{epoch + 1}, loss: {loss_both['train']}, acc :{acc_both['train']}"
            f"val loss:{loss_both['val']},val acc: {acc_both['val']}"
        )
        # 训练完一个epoch后打印： time :xx, epoch :x, loss: xx, acc :xx val loss:xx, val acc: xx
        if acc_both["val"] > best_acc:
            best_acc = acc_both["val"]
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        # 将当前epoch的训练结果与过去最好的结果进行比较，如果更好，则在对应地址下更新参数，如果没有变好，则不保存参数。

        # 写入tensorboard 供查看训练过程
        writer.add_scalars("epoch_accuracy", tag_scalar_dict=acc_both, global_step=epoch)
        writer.add_scalars("epoch_loss", tag_scalar_dict=loss_both, global_step=epoch)

        #"""——EarlyStopping插入位置，注意缩进——"""
        early_stopping(loss_both['val'], model)
        if early_stopping.early_stop:     # 判断是否满足停止条件
            print("Early stopping")
            break

    # 将训练的参数载入模型
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    return model      # 返回带训练参数的模型

# <2.3 定义功能函数与训练>
# 定义损失函数，交叉熵损失函数
criterion=nn.CrossEntropyLoss(weight=torch.FloatTensor(label_weight).to(device))
# 定义优化函数，adam优化函数
optimizer = optim.Adam(model.parameters(), lr=1e-2)
# 调整学习率，40个epoch学习率衰减0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=200)       # 调用训练函数，开始训练



