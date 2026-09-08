import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnablePositionEncoding(nn.Module):
    def __init__(self, D, W, H, head_dim):
        super(LearnablePositionEncoding, self).__init__()
        self.loc = nn.Parameter(torch.randn(1, 3, D, W, H))  # 可学习的位置编码
        self.conv_p = nn.Conv3d(3, head_dim, kernel_size=1)  # 将位置编码映射到特征空间

    def forward(self, x):
        pe = self.conv_p(self.loc)  # 生成位置编码
        return pe
    
def position(D, W, H, is_cuda=True):
    if is_cuda:
        loc_d = torch.linspace(-1.0, 1.0, D).cuda().unsqueeze(1).unsqueeze(2).repeat(1, W, H)
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).unsqueeze(2).repeat(D, 1, H)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(0).unsqueeze(1).repeat(D, W, 1)
    else:
        loc_d = torch.linspace(-1.0, 1.0, D).unsqueeze(1).unsqueeze(2).repeat(1, W, H)
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).unsqueeze(2).repeat(D, 1, H)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(0).unsqueeze(1).repeat(D, W, 1)

    loc = torch.cat([loc_d.unsqueeze(0), loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

def stride(x, stride):
    b, c, d, w, h = x.shape
    return x[:, :, ::stride, ::stride, ::stride]

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)

class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=3, head=64, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head  # 增加注意力头的数量
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.head_dim = self.out_planes // self.head

        # 定义可学习的权重参数
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))  # 可学习权重
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))  # 可学习权重

        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv3d(in_planes, out_planes, kernel_size=1)
        self.position_encoding = LearnablePositionEncoding(64, 64, 64, self.head_dim)  # 可学习的位置编码

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad3d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)
        self.norm = nn.LayerNorm([64])  # 使用 LayerNorm 对注意力权重进行归一化，normalized_shape=[64]

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv3d(self.kernel_conv * self.kernel_conv * self.kernel_conv, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化可学习权重
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)

        # 初始化卷积核
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv * self.kernel_conv):
            kernel[i, i // (self.kernel_conv * self.kernel_conv), (i // self.kernel_conv) % self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def unfold3d(self, input, kernel_size, stride=1, padding=0, dilation=1):
        # 参数标准化处理
        def to_3tuple(x):
            return (x, x, x) if isinstance(x, int) else x

        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)
        padding = to_3tuple(padding)
        dilation = to_3tuple(dilation)

        # 添加填充
        if any(p > 0 for p in padding):
            input = F.pad(input, (padding[2], padding[2],   # width 左右填充
                          padding[1], padding[1],   # height 上下填充
                          padding[0], padding[0]))  # depth 前后填充

        # 计算输出尺寸
        batch, channels, d, w, h = input.shape
        kd, kw, kh = kernel_size
        sd, sw, sh = stride
        dd, dw, dh = dilation

        # 计算有效核尺寸（考虑膨胀）
        eff_kd = (kd - 1) * dd + 1
        eff_kh = (kh - 1) * dh + 1
        eff_kw = (kw - 1) * dw + 1

        # 计算输出空间维度
        out_d = (d - eff_kd) // sd + 1
        out_h = (h - eff_kh) // sh + 1
        out_w = (w - eff_kw) // sw + 1

        # 生成展开索引
        input_unfold = input.unfold(2, kd, sd)  # [batch, C, out_d, H, W, kd]
        input_unfold = input_unfold.unfold(3, kh, sh)  # [batch, C, out_d, out_h, W, kd, kh]
        input_unfold = input_unfold.unfold(4, kw, sw)  # [batch, C, out_d, out_h, out_w, kd, kh, kw]

        # 处理膨胀（通过步长间隔采样）
        if any(d > 1 for d in dilation):
            indices_d = torch.arange(0, kd * dd, dd, device=input.device)
            indices_h = torch.arange(0, kh * dh, dh, device=input.device)
            indices_w = torch.arange(0, kw * dw, dw, device=input.device)
            input_unfold = input_unfold[:, :, :, :, :, 
                               indices_d[:kd],
                               indices_h[:kh],
                               indices_w[:kw]]

        # 调整维度顺序
        input_unfold = input_unfold.permute(0, 1, 5, 6, 7, 2, 3, 4)  # [batch, C, kd, kh, kw, out_d, out_h, out_w]

        # 合并通道和卷积核维度
        input_unfold = input_unfold.contiguous().view(batch, channels * kd * kh * kw, -1)

        return input_unfold

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, d, w, h = q.shape
        d_out, w_out, h_out = d // self.stride, w // self.stride, h // self.stride

        pe = self.position_encoding(x)  # 使用可学习的位置编码
        q_att = q.view(b * self.head, self.head_dim, d, w, h) * scaling
        k_att = k.view(b * self.head, self.head_dim, d, w, h)
        v_att = v.view(b * self.head, self.head_dim, d, w, h)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold3d(self.pad_att(k_att), 3).view(b * self.head, self.head_dim,
                                                             self.kernel_att * self.kernel_att * self.kernel_att, d_out,
                                                             w_out, h_out)
        unfold_rpe = self.unfold3d(self.pad_att(pe), 3).view(1, self.head_dim, self.kernel_att * self.kernel_att * self.kernel_att, d_out,
                                                            w_out, h_out)

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = self.norm(att)  # 对注意力权重进行归一化
        att = self.softmax(att)

        out_att = self.unfold3d(self.pad_att(v_att), 3).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att * self.kernel_att,
                                                            d_out, w_out, h_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, d_out, w_out, h_out)

        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, d * w * h), k.view(b, self.head, self.head_dim, d * w * h),
             v.view(b, self.head, self.head_dim, d * w * h)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[2], x.shape[3], x.shape[4])

        out_conv = self.dep_conv(f_conv)
        rate1 = torch.rand(1, device=x.device)  # 动态生成权重
        rate2 = 1.0 - rate1  # 确保权重之和为 1
        return rate1 * out_att + rate2 * out_conv

if __name__ == '__main__':
    block = ACmix(in_planes=64, out_planes=64)
    input = torch.rand(1, 64, 8, 64, 64)
    output = block(input)
    print(output.shape)

'''
    def unfold3d(self, input, kernel_size, stride=1, padding=0, dilation=1):
        input_padded = F.pad(input, (padding, padding, padding, padding, padding, padding))

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        N, C, D, H, W = input.shape
        kD, kH, kW = kernel_size
        sD, sH, sW = stride
        pD, pH, pW = padding
        dD, dH, dW = dilation

        D_out = (D + 2 * pD - dD * (kD - 1) - 1) // sD + 1
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

        unfolded_output = torch.zeros(N, C * kD * kH * kW, D_out * H_out * W_out, device=input.device)

        count = 0
        for d in range(0, D - kD + 1 + 2 * pD, sD):
            for h in range(0, H - kH + 1 + 2 * pH, sH):
                for w in range(0, W - kW + 1 + 2 * pW, sW):
                    patch = input_padded[:, :, d:d + kD, h:h + kH, w:w + kW]
                    unfolded_output[:, :, count] = patch.reshape(N, C * kD * kH * kW)
                    count += 1

        return unfolded_output
    '''