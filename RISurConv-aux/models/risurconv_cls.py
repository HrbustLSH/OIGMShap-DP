import torch.nn as nn
import torch.nn.functional as F
from models.risurconv_utils import RISurConvSetAbstraction

class get_model(nn.Module):
    def __init__(self,num_class, n, normal_channel=True):
        super(get_model, self).__init__()
        self.normal_channel = normal_channel
        self.sc0 = RISurConvSetAbstraction(npoint=512*n, radius=0.12, nsample=8, in_channel= 0, out_channel=32, group_all=False)
        self.sc1 = RISurConvSetAbstraction(npoint=256*n, radius=0.16, nsample=16, in_channel=32, out_channel=64,  group_all=False)
        self.sc2 = RISurConvSetAbstraction(npoint=128*n, radius=0.24, nsample=32, in_channel=64, out_channel=128,  group_all=False)
        self.sc3 = RISurConvSetAbstraction(npoint=64*n, radius=0.48, nsample=64, in_channel=128, out_channel=256,  group_all=False)
        self.sc4 = RISurConvSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, out_channel=512,  group_all=True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8,dropout=0.05)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, :, 3:]
            xyz = xyz[:, :, :3]
        else:
            # compute the LRA and use as normal
            norm = None

        l0_xyz, l0_norm, l0_points = self.sc0(xyz, norm, None)
        l1_xyz, l1_norm, l1_points = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sc3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points = self.sc4(l3_xyz, l3_norm, l3_points)

        x=l4_points.permute(0, 2, 1)
        x=self.transformer_encoder(x)
        globle_x = x.view(B, 512)
        # globle_x = torch.max(l4_feature, 2)[0]
        
        x = self.drop1(F.relu(self.bn1(self.fc1(globle_x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        
        return x, l4_points

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss

class get_model_aux(nn.Module):
    def __init__(self, args, output_channels=40):
        super(get_model_aux, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # 定义辅助分支的1x1卷积层
        self.aux_conv = nn.Conv1d(64, args.emb_dims, kernel_size=1, bias=False)
        self.aux_bn = nn.BatchNorm1d(args.emb_dims)
        self.aux_relu = nn.ReLU()
        
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        # print(f'self.k:{self.k}')
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        
        
        
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
       
        # 在第一个卷积层之后添加辅助分支
        #if self.training :
        aux_features = self.aux_relu(self.aux_bn(self.aux_conv(x2))) 
        aux_features = F.adaptive_avg_pool1d(aux_features, 1).view(batch_size, -1) #10,1024

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        # print(f'x.shape:{x.shape}')#10,1024,1024
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)#10,1024
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # print(f'x1.shape:{x1.shape}')
        x = torch.cat((x1, x2), 1) #10,2048
        
        # print(f'aux_features.shape:{aux_features.shape}')
        # 将辅助分支的输出与主模型的输出拼接
        x = torch.cat((x1, aux_features), 1) #11,3072 
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        # print(f'x.shape:{x.shape}')
        return x