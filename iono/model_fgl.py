import random
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """ConvLSTM 核心计算单元"""
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        )


class PrivGRU(nn.Module):
    """轻量 GRU，将 aux_future 序列聚合为时序感知的向量"""
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, aux_seq):
        """
        aux_seq: (B, T, input_dim)
        输出: (B, hidden_dim)
        """
        _, h_n = self.gru(aux_seq)  # h_n: (1, B, hidden_dim)
        return h_n.squeeze(0)       # (B, hidden_dim)

class SpatialFiLM(nn.Module):
    """空间自适应 FiLM 层：保留空间维度，通过 1x1 卷积生成逐像素的调制参数"""
    def __init__(self, condition_channels, hidden_channels):
        super().__init__()
        # 使用 1x1 卷积代替 Linear 层
        self.conv = nn.Sequential(
            nn.Conv2d(condition_channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2 * hidden_channels, kernel_size=1) # 输出 gamma 和 beta
        )
        
        # 零初始化：保证训练初期 gamma=0, beta=0，即 h * 1 + 0 = h
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, h, condition_spatial):
        """
        h: (B, hidden_channels, H, W) — 历史 ConvLSTM 的隐状态
        condition_spatial: (B, condition_channels, H, W) — 保留了空间的未来特权特征
        """
        gamma_beta = self.conv(condition_spatial)              # (B, 2*C, H, W)
        gamma, beta = gamma_beta.chunk(2, dim=1)               # 各自 (B, C, H, W)
        
        return h * (1.0 + gamma) + beta

class StudentForecaster(nn.Module):
    """
    FGL 时空序列预测学生模型 (Student)
    
    架构：
    - 历史编码器：多层 ConvLSTM，编码历史 TEC + 辅助参数
    - PrivGRU：轻量 GRU，聚合未来辅助参数的时序信息 (非特权信息)
    - FiLM：利用未来辅助参数对历史隐状态进行全局空间调制
    - 解码器：多层 ConvLSTM，自回归生成多步预测
    
    特点：没有未来 TEC 的“上帝视角”输入，纯粹利用已知/预估特征。
    """
    def __init__(self, in_channels=1, hidden_channels=64, num_layers=2,
                 num_aux=0, pred_steps=24, priv_gru_hidden=32):
        super().__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_aux = num_aux
        self.pred_steps = pred_steps

        # --- 历史编码器 ---
        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels + num_aux if i == 0 else hidden_channels
            self.encoder_cells.append(ConvLSTMCell(in_ch, hidden_channels))

        # --- 未来辅助信息编码与调制机制 ---
        self.priv_gru = PrivGRU(input_dim=num_aux, hidden_dim=priv_gru_hidden)
        
        self.aux_film_layers = nn.ModuleList([
            SpatialFiLM(priv_gru_hidden, hidden_channels)
            for _ in range(num_layers)
        ])

        # --- 解码器 ---
        self.decoder_cells = nn.ModuleList()
        for i in range(num_layers):
            decoder_in = in_channels + num_aux if i == 0 else hidden_channels
            self.decoder_cells.append(ConvLSTMCell(decoder_in, hidden_channels))

        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def _build_input(self, base_input, aux_t, height, width):
        if self.num_aux <= 0: return base_input
        aux_map = aux_t.view(base_input.size(0), self.num_aux, 1, 1).expand(-1, -1, height, width)
        return torch.cat([base_input, aux_map], dim=1)

    def _encode_sequence(self, tec_map, aux_x, encoder_cells):
        B, T, _, H, W = tec_map.shape
        hidden_states = [cell.init_hidden(B, H, W, tec_map.device) for cell in encoder_cells]
        for t in range(T):
            input_t = self._build_input(tec_map[:, t], aux_x[:, t], H, W)
            for i, cell in enumerate(encoder_cells):
                if i == 0: hidden_states[i] = cell(input_t, hidden_states[i])
                else: hidden_states[i] = cell(hidden_states[i-1][0], hidden_states[i])
        return hidden_states

    def _decode_step(self, dec_input, dec_states):
        for i, cell in enumerate(self.decoder_cells):
            h_prev, c_prev = dec_states[i]
            if i == 0: dec_states[i] = cell(dec_input, (h_prev, c_prev))
            else: dec_states[i] = cell(dec_states[i-1][0], (h_prev, c_prev))
        return self.head(dec_states[-1][0])

    def forward(self, tec_map, aux_x=None,
                future_aux=None, dec_aux=None, 
                return_hidden=False, y_true=None, tf_ratio=None):
        B, _, _, H, W = tec_map.shape

        # 1. 编码历史序列
        hist_states = self._encode_sequence(tec_map, aux_x, self.encoder_cells)

        # 2. 未来辅助信息调制
        fused_states = []
        h_aux_spatial = None
        if future_aux is not None:
            h_aux = self.priv_gru(future_aux)
            h_aux_spatial = h_aux.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        for i, (h, c) in enumerate(hist_states):
            h_mod = h
            if h_aux_spatial is not None:
                h_mod = self.aux_film_layers[i](h_mod, h_aux_spatial)
            fused_states.append((h_mod, c))

        # 3. 解码流程
        decoder_states = [(h, c) for h, c in fused_states]
        outputs = []
        base = tec_map[:, -1]
        
        spatial_grid = base[:, 1:] if base.shape[1] > 1 else None

        for step in range(self.pred_steps):
            if dec_aux is not None: aux_step = dec_aux[:, step]
            else: aux_step = torch.zeros(B, self.num_aux, device=tec_map.device)
            
            dec_input = self._build_input(base, aux_step, H, W)
            pred_frame = self._decode_step(dec_input, decoder_states)
            outputs.append(pred_frame)
            
            use_tf = (self.training and y_true is not None and tf_ratio is not None and random.random() < tf_ratio)
            next_tec = y_true[:, step] if use_tf else pred_frame
            base = torch.cat([next_tec, spatial_grid], dim=1) if spatial_grid is not None else next_tec

        output = torch.stack(outputs, dim=1)
        
        if return_hidden:
            return output, [state[0] for state in fused_states]
        return output

class TeacherForecaster(nn.Module):
    """
    FGL 时空序列预测教师模型 (Teacher)
    
    架构：在 Student 的基础上增加了针对 future_tec (上帝视角特权) 的双重 FiLM 调制管道。
    """
    def __init__(self, in_channels=1, hidden_channels=64, num_layers=2,
                 num_aux=0, pred_steps=24, priv_gru_hidden=32):
        super().__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_aux = num_aux
        self.pred_steps = pred_steps

        # --- 历史编码器 ---
        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels + num_aux if i == 0 else hidden_channels
            self.encoder_cells.append(ConvLSTMCell(in_ch, hidden_channels))

        # --- 特权编码与未来辅助信息编码 ---
        self.privileged_encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels + num_aux if i == 0 else hidden_channels
            self.privileged_encoder_cells.append(ConvLSTMCell(in_ch, hidden_channels))

        self.priv_gru = PrivGRU(input_dim=num_aux, hidden_dim=priv_gru_hidden)

        # --- 双管 FiLM ---
        fused_condition_dim = priv_gru_hidden + hidden_channels
        self.film_layers = nn.ModuleList([
            SpatialFiLM(fused_condition_dim, hidden_channels) for _ in range(num_layers)
        ])

        # --- 解码器 ---
        self.decoder_cells = nn.ModuleList()
        for i in range(num_layers):
            decoder_in = in_channels + num_aux if i == 0 else hidden_channels
            self.decoder_cells.append(ConvLSTMCell(decoder_in, hidden_channels))

        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def _build_input(self, base_input, aux_t, height, width):
        if self.num_aux <= 0: return base_input
        aux_map = aux_t.view(base_input.size(0), self.num_aux, 1, 1).expand(-1, -1, height, width)
        return torch.cat([base_input, aux_map], dim=1)

    def _encode_sequence(self, tec_map, aux_x, encoder_cells):
        B, T, _, H, W = tec_map.shape
        hidden_states = [cell.init_hidden(B, H, W, tec_map.device) for cell in encoder_cells]
        for t in range(T):
            input_t = self._build_input(tec_map[:, t], aux_x[:, t], H, W)
            for i, cell in enumerate(encoder_cells):
                if i == 0: hidden_states[i] = cell(input_t, hidden_states[i])
                else: hidden_states[i] = cell(hidden_states[i-1][0], hidden_states[i])
        return hidden_states

    def _decode_step(self, dec_input, dec_states):
        for i, cell in enumerate(self.decoder_cells):
            h_prev, c_prev = dec_states[i]
            if i == 0: dec_states[i] = cell(dec_input, (h_prev, c_prev))
            else: dec_states[i] = cell(dec_states[i-1][0], (h_prev, c_prev))
        return self.head(dec_states[-1][0])

    def forward(self, tec_map, aux_x=None,
                future_tec=None, future_aux=None,
                dec_aux=None, return_hidden=False,
                y_true=None, tf_ratio=None):
        B, _, _, H, W = tec_map.shape

        # 1. 编码历史序列
        hist_states = self._encode_sequence(tec_map, aux_x, self.encoder_cells)

        # 2. 双重调制 (优先 aux，后 tec)
        h_aux_spatial, future_states = None, None
        
        if future_aux is not None:
            h_aux = self.priv_gru(future_aux)
            h_aux_spatial = h_aux.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        if future_tec is not None:
            future_states = self._encode_sequence(future_tec, future_aux, self.privileged_encoder_cells)

        fused_states = []
        for i, (h, c) in enumerate(hist_states):
            h_mod = h
            if h_aux_spatial is not None and future_states is not None:
                # 1. 提取特权 TEC 的隐状态
                priv_h = future_states[i][0]
                # 2. 在通道维度拼接：[B, 64, H, W] 和 [B, 32, H, W] -> [B, 96, H, W]
                fused_condition = torch.cat([priv_h, h_aux_spatial], dim=1)
                # 3. 进行单次、安全、全局的 FiLM 调制
                h_mod = self.film_layers[i](h_mod, fused_condition)
            fused_states.append((h_mod, c))

        # 3. 解码流程
        decoder_states = [(h, c) for h, c in fused_states]
        outputs = []
        base = tec_map[:, -1]
        
        spatial_grid = base[:, 1:] if base.shape[1] > 1 else None

        for step in range(self.pred_steps):
            if dec_aux is not None: aux_step = dec_aux[:, step]
            else: aux_step = torch.zeros(B, self.num_aux, device=tec_map.device)
            
            dec_input = self._build_input(base, aux_step, H, W)
            pred_frame = self._decode_step(dec_input, decoder_states)
            outputs.append(pred_frame)
            
            use_tf = (self.training and y_true is not None and tf_ratio is not None and random.random() < tf_ratio)
            next_tec = y_true[:, step] if use_tf else pred_frame
            base = torch.cat([next_tec, spatial_grid], dim=1) if spatial_grid is not None else next_tec

        output = torch.stack(outputs, dim=1)
        
        if return_hidden:
            return output, [state[0] for state in fused_states]
        return output