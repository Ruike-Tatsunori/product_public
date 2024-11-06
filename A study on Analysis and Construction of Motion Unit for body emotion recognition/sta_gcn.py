import numpy as np
import IPython
from IPython import display
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import torch
import torch.nn as nn
import torch.nn.functional as F

print('Use CUDA:', torch.cuda.is_available())

seed = 123
# Numpy
np.random.seed(seed)
# Pytorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

class Feeder(torch.utils.data.Dataset):
  def __init__(self, data_path, label_path):
      super().__init__()
      self.label = np.load(label_path)
      self.data = np.load(data_path)

  def __len__(self):
      return len(self.label)

  def __iter__(self):
      return self

  def __getitem__(self, index):
      data = np.array(self.data[index])
      label = self.label[index]

      return data, label
  
class Graph():
  def __init__(self, hop_size=2):
    self.get_edge()
    self.hop_size = hop_size 
    self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)
    self.get_adjacency() 

  def __str__(self):
    return self.A

  def get_edge(self):
    self.num_node = 24
    self_link = [(i, i) for i in range(self.num_node)] # ループ
    neighbor_base = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                     (6, 7), (7, 8), (6, 9), (9, 10), (10, 11),
                     (11, 12), (6, 13), (13, 14), (14, 15), (15, 16),
                     (1, 17), (17, 18), (18, 19), (19, 20), (1, 21),
                     (21, 22), (22, 23), (23, 24)]
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
    self.edge = self_link + neighbor_link

  def get_adjacency(self):
    valid_hop = range(0, self.hop_size + 1, 1)
    adjacency = np.zeros((self.num_node, self.num_node))
    for hop in valid_hop:
        adjacency[self.hop_dis == hop] = 1
    normalize_adjacency = self.normalize_digraph(adjacency)
    A = np.zeros((len(valid_hop), self.num_node, self.num_node))
    for i, hop in enumerate(valid_hop):
        A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
    self.A = A

  def get_hop_distance(self, num_node, edge, hop_size):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(hop_size, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

  def normalize_digraph(self, A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    DAD = np.dot(A, Dn)
    return DAD
  
class S_GC(nn.Module):
  def __init__(self, in_channels, out_channels, s_kernel_size):
    super(S_GC, self).__init__()
    self.s_kernel_size = s_kernel_size
    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels * s_kernel_size,
                          kernel_size=1)
    
  def forward(self, x, A, att_edge=None):
    x = self.conv(x)
    n, kc, t, v = x.size()
    x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
    x = torch.einsum('nkctv,kvw->nctw', (x, A))
    return x.contiguous()
  
class S_GC_att_edge(nn.Module):
  def __init__(self, in_channels, out_channels, s_kernel_size, num_att_edge):
    super(S_GC_att_edge, self).__init__()
    self.num_att_edge = num_att_edge
    self.s_kernel_size = s_kernel_size + num_att_edge
    self.conv = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels * self.s_kernel_size,
                           kernel_size=1)

  def forward(self, x, A, att_edge):
    x = self.conv(x)
    n, kc, t, v = x.size()
    x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
    x1 = x[:, :self.s_kernel_size-self.num_att_edge, :, :, :]
    x2 = x[:, -self.num_att_edge:, :, :, :]
    x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))
    x2 = torch.einsum('nkctv,nkvw->nctw', (x2, att_edge))
    x_sum = x1 + x2

    return x_sum
  
class STGC_Block(nn.Module):
  def __init__(self, in_channels, out_channels, stride, s_kernel_size, t_kernel_size, dropout, A_size, num_att_edge=0, use_att_edge=False):
    super(STGC_Block, self).__init__()
    # 空間グラフ畳み込み attention edgeありかなしか
    if not use_att_edge:
      self.sgc = S_GC(in_channels=in_channels,
                       out_channels=out_channels,
                       s_kernel_size=s_kernel_size)
    else:
      self.sgc = S_GC_att_edge(in_channels=in_channels,
                                 out_channels=out_channels,
                                 s_kernel_size=s_kernel_size,
                                 num_att_edge=num_att_edge)

    # Learnable weight matrix M エッジに重みを与えます. どのエッジが重要かを学習します.
    self.M = nn.Parameter(torch.ones(A_size))
    
    # 時間グラフ畳み込み
    self.tgc = nn.Sequential(nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Conv2d(out_channels,
                                      out_channels,
                                      (t_kernel_size, 1),
                                      (stride, 1),
                                      ((t_kernel_size - 1) // 2, 0)),
                            nn.BatchNorm2d(out_channels),
                            nn.Dropout(dropout),
                            nn.ReLU()) 
    
    # 残差処理
    if(in_channels == out_channels) and (stride == 1):
      self.residual = lambda x: x
    else:
      self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              stride=(stride, 1)),
                                    nn.BatchNorm2d(out_channels))

  def forward(self, x, A, att_edge):
    x = self.tgc(self.sgc(x, A * self.M, att_edge)) + self.residual(x)
    return x
  
class FeatureExtractor(nn.Module):
  def __init__(self, config, s_kernel_size, t_kernel_size, dropout, A_size):
    super(FeatureExtractor, self).__init__()
    # Batch Normalization
    self.bn = nn.BatchNorm1d(config[0][0] * A_size[2])

    # STGC-Block config
    kwargs = dict(s_kernel_size=s_kernel_size,
                  t_kernel_size=t_kernel_size,
                  dropout=dropout,
                  A_size=A_size)
    self.stgc_block1 = STGC_Block(config[0][0], config[0][1], config[0][2], **kwargs)
    self.stgc_block2 = STGC_Block(config[1][0], config[1][1], config[1][2], **kwargs)
    self.stgc_block3 = STGC_Block(config[2][0], config[2][1], config[2][2], **kwargs)

  def forward(self, x, A):
    # Batch Normalization
    N, C, T, V = x.size() # batch, channel, frame, node
    x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
    x = self.bn(x)
    x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
    # STGC Blocks
    x = self.stgc_block1(x, A, None)
    x = self.stgc_block2(x, A, None)
    x = self.stgc_block3(x, A, None)
    return x
  
class AttentionBranch(nn.Module):
  def __init__(self, config, num_classes, num_att_edge, s_kernel_size, t_kernel_size, dropout, A_size):
    super(AttentionBranch, self).__init__()
    # STGC-Block config
    kwargs = dict(s_kernel_size=s_kernel_size,
                  t_kernel_size=t_kernel_size,
                  dropout=dropout,
                  A_size=A_size)
    self.stgc_block1 = STGC_Block(config[0][0], config[0][1], config[0][2], **kwargs)
    self.stgc_block2 = STGC_Block(config[1][0], config[1][1], config[1][2], **kwargs)
    self.stgc_block3 = STGC_Block(config[2][0], config[2][1], config[2][2], **kwargs)

    # Prediction
    self.fc = nn.Conv2d(config[-1][1], num_classes, kernel_size=1, padding=0)

    # Attention
    self.att_bn = nn.BatchNorm2d(config[-1][1])
    self.att_conv = nn.Conv2d(config[-1][1], num_classes, kernel_size=1, padding=0, stride=1, bias=False)

    # Attention node
    self.att_node_conv = nn.Conv2d(num_classes, 1, kernel_size=1, padding=0, stride=1, bias=False)
    self.att_node_bn = nn.BatchNorm2d(1)
    self.sigmoid = nn.Sigmoid()

    # Attention edge
    self.num_att_edge = num_att_edge
    self.att_edge_conv = nn.Conv2d(num_classes, num_att_edge * A_size[2], kernel_size=1, padding=0, stride=1, bias=False)
    self.att_edge_bn = nn.BatchNorm2d(num_att_edge * A_size[2])
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()

  def forward(self, x, A):
    N, c, T, V = x.size()

    # STGC Block
    x = self.stgc_block1(x, A, None)
    x = self.stgc_block2(x, A, None)
    x = self.stgc_block3(x, A, None)

    # Prediction
    x_out = F.avg_pool2d(x, x.size()[2:])
    x_out = x_out.view(N, -1, 1, 1)
    x_out = self.fc(x_out)
    output = x_out.view(x_out.size(0), -1)

    # Attention
    x_att = self.att_bn(x)
    x_att = self.att_conv(x_att)

    # Attention node
    x_node = self.att_node_conv(x_att)
    x_node = self.att_node_bn(x_node)
    x_node = F.interpolate(x_node, size=(T, V))
    att_node = self.sigmoid(x_node)

    # Attention edge
    x_edge = F.avg_pool2d(x_att, (x_att.size()[2], 1))
    x_edge = self.att_edge_conv(x_edge)
    x_edge = self.att_edge_bn(x_edge)
    x_edge = x_edge.view(N, self.num_att_edge, V, V)
    x_edge = self.tanh(x_edge)
    att_edge = self.relu(x_edge)

    return output, att_node, att_edge
  
class PerceptionBranch(nn.Module):
  def __init__(self, config, num_classes, num_att_edge, s_kernel_size, t_kernel_size, dropout, A_size, use_att_edge=True):
    super(PerceptionBranch, self).__init__()
    # STGC-Block config
    kwargs = dict(s_kernel_size=s_kernel_size,
                  t_kernel_size=t_kernel_size,
                  dropout=dropout,
                  A_size=A_size,
                  num_att_edge=num_att_edge,
                  use_att_edge=use_att_edge)
    self.stgc_block1 = STGC_Block(config[0][0], config[0][1], config[0][2], **kwargs)
    self.stgc_block2 = STGC_Block(config[1][0], config[1][1], config[1][2], **kwargs)
    self.stgc_block3 = STGC_Block(config[2][0], config[2][1], config[2][2], **kwargs)

    # Prediction
    self.fc = nn.Conv2d(config[-1][1], num_classes, kernel_size=1, padding=0)

  def forward(self, x, A, att_edge):
    N, c, T, V = x.size()
    # STGC Block
    x = self.stgc_block1(x, A, att_edge)
    x = self.stgc_block2(x, A, att_edge)
    x = self.stgc_block3(x, A, att_edge)

    # Prediction
    x = F.avg_pool2d(x, x.size()[2:])
    x = x.view(N, -1, 1, 1)
    x = self.fc(x)
    output = x.view(x.size(0), -1)

    return output
  
class STA_GCN(nn.Module):
  def __init__(self, num_classes, in_channels, t_kernel_size, hop_size, num_att_edge, dropout=0.5):
    super(STA_GCN, self).__init__()

    # Graph
    graph = Graph(hop_size)
    A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
    self.register_buffer('A', A)

    kwargs = dict(s_kernel_size=A.size(0),
                   t_kernel_size=t_kernel_size,
                   dropout=dropout,
                   A_size=A.size())

    # Feature extractor
    f_config = [[in_channels, 32, 1], [32, 32, 1], [32, 32, 1]]
    self.feature_extractor = FeatureExtractor(f_config, **kwargs)

    # Attention branch
    a_config = [[32, 64, 2], [64, 64, 1], [64, 64, 1]]
    self.attention_branch = AttentionBranch(a_config, num_classes, num_att_edge, **kwargs)

    # Perception branch
    p_config = [[32, 64, 2], [64, 64, 1], [64, 64, 1]]
    self.perception_branch = PerceptionBranch(p_config, num_classes, num_att_edge, **kwargs)

  def forward(self, x):
    # Feature extractor
    feature = self.feature_extractor(x, self.A)

    # Attention branch
    output_ab, att_node, att_edge = self.attention_branch(feature, self.A)

    # Attention mechanism
    att_x = feature * att_node
    
    # Perception branch
    output_pb = self.perception_branch(att_x, self.A, att_edge)

    return output_ab, output_pb, att_node, att_edge
  
NUM_EPOCH = 100
BATCH_SIZE = 64
HOP_SIZE = 2
NUM_ATT_EDGE = 2 # 動作ごとのattention edgeの生成数

# モデルを作成
model = STA_GCN(num_classes=12, 
                  in_channels=3,
                  t_kernel_size=9, # 時間グラフ畳み込みのカーネルサイズ (t_kernel_size × 1)
                  hop_size=HOP_SIZE,
                  num_att_edge=NUM_ATT_EDGE).cuda()

# オプティマイザ
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 誤差関数
criterion = torch.nn.CrossEntropyLoss()

# データセットの用意
data_loader = dict()
data_loader['train'] = torch.utils.data.DataLoader(dataset=Feeder(data_path='data/train_data.npy', label_path='data/train_label.npy'), batch_size=BATCH_SIZE, shuffle=True,)
data_loader['test'] = torch.utils.data.DataLoader(dataset=Feeder(data_path='data/test_data.npy', label_path='data/test_label.npy'), batch_size=BATCH_SIZE, shuffle=False)

# モデルを学習モードに変更
model.train()

# 学習開始
for epoch in range(1, NUM_EPOCH+1):
  correct_pb = 0
  sum_loss = 0
  for batch_idx, (data, label) in enumerate(data_loader['train']):
    data = data.cuda()
    label = label.cuda()

    output_ab, output_pb, _, _ = model(data)

    loss = criterion(output_ab, label) + criterion(output_pb, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    sum_loss += loss.item()
    _, predict = torch.max(output_pb.data, 1)
    correct_pb += (predict == label).sum().item()

  print('# Epoch: {} | Loss: {:.4f} | Accuracy PB: {:.3f}[%]'.format(epoch, sum_loss/len(data_loader['train'].dataset), (100. * correct_pb / len(data_loader['train'].dataset))))

# モデルを評価モードに変更
model.eval()

correct_pb = 0
confusion_matrix = np.zeros((12, 12))
with torch.no_grad():
  for batch_idx, (data, label) in enumerate(data_loader['test']):
    data = data.cuda()
    label = label.cuda()

    output_ab, output_pb, _, _ = model(data)

    _, predict = torch.max(output_pb.data, 1)
    correct_pb += (predict == label).sum().item()

    for l, p in zip(label.view(-1), predict.view(-1)):
      confusion_matrix[l.long(), p.long()] += 1

len_cm = len(confusion_matrix)
for i in range(len_cm):
    sum_cm = np.sum(confusion_matrix[i])
    for j in range(len_cm):
        confusion_matrix[i][j] = 100 * (confusion_matrix[i][j] / sum_cm)

emotion = ["anger", "contempt", "disgust", "fear", "gratitude", "guilt", "jealousy", "joy", "pride", "sadness", "shame", "surprise"]
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.tight_layout()
tick_marks = np.arange(len(emotion))
plt.xticks(tick_marks, emotion, rotation=45)
plt.yticks(tick_marks, emotion)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

print('# Test Accuracy PB: {:.3f}[%]'.format((100. * correct_pb / len(data_loader['test'].dataset))))