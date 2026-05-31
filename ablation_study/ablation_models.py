"""
消融实验模型定义
验证 FGL（Future Guidance Learning）和 FiLM（Feature-wise Linear Modulation）的独立贡献与协同效应
"""

import random
import torch
import torch.nn as nn
from iono.model_fgl import ConvLSTMCell, PrivGRU, SpatialFiLM


class BaselineModel(nn.Module):
    """
    w/o Both: 基线模型
    - 纯粹的 ConvLSTM 编码器-解码器架构，使用了aux历史数据拼接
    - 没有 FGL（知识蒸馏），没有 FiLM（特征调制）
    - 历史 TEC 直接编码，解码器自回归预测
    """
    def __init__(self, in_channels=1, hidden_channels=64, num_layers=2,
                 num_aux=0, pred_steps=24, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_aux = num_aux
        self.pred_steps = pred_steps

        # 历史编码器
        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels + num_aux if i == 0 else hidden_channels
            self.encoder_cells.append(ConvLSTMCell(in_ch, hidden_channels))

        # 解码器
        self.decoder_cells = nn.ModuleList()
        for i in range(num_layers):
            decoder_in = in_channels + num_aux if i == 0 else hidden_channels
            self.decoder_cells.append(ConvLSTMCell(decoder_in, hidden_channels))

        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def _build_input(self, base_input, aux_t, height, width):
        """融合 TEC 和辅助特征"""
        if self.num_aux <= 0:
            return base_input
        aux_map = aux_t.view(base_input.size(0), self.num_aux, 1, 1).expand(-1, -1, height, width)
        return torch.cat([base_input, aux_map], dim=1)

    def _encode_sequence(self, tec_map, aux_x):
        """编码历史序列"""
        B, T, _, H, W = tec_map.shape
        hidden_states = [cell.init_hidden(B, H, W, tec_map.device) for cell in self.encoder_cells]
        
        for t in range(T):
            input_t = self._build_input(tec_map[:, t], aux_x[:, t], H, W)
            for i, cell in enumerate(self.encoder_cells):
                if i == 0:
                    hidden_states[i] = cell(input_t, hidden_states[i])
                else:
                    hidden_states[i] = cell(hidden_states[i-1][0], hidden_states[i])
        
        return hidden_states

    def _decode_step(self, dec_input, dec_states):
        """单步解码"""
        for i, cell in enumerate(self.decoder_cells):
            h_prev, c_prev = dec_states[i]
            if i == 0:
                dec_states[i] = cell(dec_input, (h_prev, c_prev))
            else:
                dec_states[i] = cell(dec_states[i-1][0], (h_prev, c_prev))
        return self.head(dec_states[-1][0])

    def forward(self, tec_map, aux_x=None, future_aux=None, dec_aux=None,
                return_hidden=False, y_true=None, tf_ratio=None):
        B, _, _, H, W = tec_map.shape

        # 编码历史序列
        hist_states = self._encode_sequence(tec_map, aux_x)

        # 解码
        decoder_states = [(h, c) for h, c in hist_states]
        outputs = []
        base = tec_map[:, -1]
        spatial_grid = base[:, 1:] if base.shape[1] > 1 else None

        for step in range(self.pred_steps):
            aux_step = dec_aux[:, step] if dec_aux is not None else torch.zeros(B, self.num_aux, device=tec_map.device)
            dec_input = self._build_input(base, aux_step, H, W)
            pred_frame = self._decode_step(dec_input, decoder_states)
            outputs.append(pred_frame)

            # Teacher forcing
            use_tf = (self.training and y_true is not None and tf_ratio is not None 
                      and random.random() < tf_ratio)
            next_tec = y_true[:, step] if use_tf else pred_frame
            base = torch.cat([next_tec, spatial_grid], dim=1) if spatial_grid is not None else next_tec

        output = torch.stack(outputs, dim=1)
        
        if return_hidden:
            return output, [state[0] for state in hist_states]
        return output


class NoFGLModel(nn.Module):
    """
    w/o FGL: 只有 FiLM，没有知识蒸馏
    - 具有 SpatialFiLM 调制机制（使用辅助特征）
    - 没有知识蒸馏（直接从头训练，不模仿教师）
    - 学生独立学习使用辅助信息进行特征调制
    """
    def __init__(self, in_channels=1, hidden_channels=64, num_layers=2,
                 num_aux=0, pred_steps=24, priv_gru_hidden=32, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_aux = num_aux
        self.pred_steps = pred_steps

        # 历史编码器
        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels + num_aux if i == 0 else hidden_channels
            self.encoder_cells.append(ConvLSTMCell(in_ch, hidden_channels))

        # FiLM 调制：处理未来辅助信息
        self.priv_gru = PrivGRU(input_dim=num_aux, hidden_dim=priv_gru_hidden)
        self.aux_film_layers = nn.ModuleList([
            SpatialFiLM(priv_gru_hidden, hidden_channels)
            for _ in range(num_layers)
        ])

        # 解码器
        self.decoder_cells = nn.ModuleList()
        for i in range(num_layers):
            decoder_in = in_channels + num_aux if i == 0 else hidden_channels
            self.decoder_cells.append(ConvLSTMCell(decoder_in, hidden_channels))

        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def _build_input(self, base_input, aux_t, height, width):
        if self.num_aux <= 0:
            return base_input
        aux_map = aux_t.view(base_input.size(0), self.num_aux, 1, 1).expand(-1, -1, height, width)
        return torch.cat([base_input, aux_map], dim=1)

    def _encode_sequence(self, tec_map, aux_x):
        B, T, _, H, W = tec_map.shape
        hidden_states = [cell.init_hidden(B, H, W, tec_map.device) for cell in self.encoder_cells]
        
        for t in range(T):
            input_t = self._build_input(tec_map[:, t], aux_x[:, t], H, W)
            for i, cell in enumerate(self.encoder_cells):
                if i == 0:
                    hidden_states[i] = cell(input_t, hidden_states[i])
                else:
                    hidden_states[i] = cell(hidden_states[i-1][0], hidden_states[i])
        
        return hidden_states

    def _decode_step(self, dec_input, dec_states):
        for i, cell in enumerate(self.decoder_cells):
            h_prev, c_prev = dec_states[i]
            if i == 0:
                dec_states[i] = cell(dec_input, (h_prev, c_prev))
            else:
                dec_states[i] = cell(dec_states[i-1][0], (h_prev, c_prev))
        return self.head(dec_states[-1][0])

    def forward(self, tec_map, aux_x=None, future_aux=None, dec_aux=None,
                return_hidden=False, y_true=None, tf_ratio=None):
        B, _, _, H, W = tec_map.shape

        # 编码历史序列
        hist_states = self._encode_sequence(tec_map, aux_x)

        # FiLM 调制（使用未来辅助信息）
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

        # 解码
        decoder_states = [(h, c) for h, c in fused_states]
        outputs = []
        base = tec_map[:, -1]
        spatial_grid = base[:, 1:] if base.shape[1] > 1 else None

        for step in range(self.pred_steps):
            aux_step = dec_aux[:, step] if dec_aux is not None else torch.zeros(B, self.num_aux, device=tec_map.device)
            dec_input = self._build_input(base, aux_step, H, W)
            pred_frame = self._decode_step(dec_input, decoder_states)
            outputs.append(pred_frame)

            use_tf = (self.training and y_true is not None and tf_ratio is not None 
                      and random.random() < tf_ratio)
            next_tec = y_true[:, step] if use_tf else pred_frame
            base = torch.cat([next_tec, spatial_grid], dim=1) if spatial_grid is not None else next_tec

        output = torch.stack(outputs, dim=1)
        
        if return_hidden:
            return output, [state[0] for state in fused_states]
        return output


class NoFiLMModel(nn.Module):
    """
    w/o FiLM: 只有 FGL，没有 FiLM 调制
    - 有知识蒸馏（FGL）机制
    - 无 FiLM 特征调制
    - 使用早期融合（Early Fusion）：辅助特征直接与 TEC 拼接
    """
    def __init__(self, in_channels=1, hidden_channels=64, num_layers=2,
                 num_aux=0, pred_steps=24, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_aux = num_aux
        self.pred_steps = pred_steps

        # 历史编码器
        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels + num_aux if i == 0 else hidden_channels
            self.encoder_cells.append(ConvLSTMCell(in_ch, hidden_channels))

        # 解码器
        self.decoder_cells = nn.ModuleList()
        for i in range(num_layers):
            decoder_in = in_channels + num_aux if i == 0 else hidden_channels
            self.decoder_cells.append(ConvLSTMCell(decoder_in, hidden_channels))

        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def _build_input(self, base_input, aux_t, height, width):
        if self.num_aux <= 0:
            return base_input
        aux_map = aux_t.view(base_input.size(0), self.num_aux, 1, 1).expand(-1, -1, height, width)
        return torch.cat([base_input, aux_map], dim=1)

    def _encode_sequence(self, tec_map, aux_x):
        B, T, _, H, W = tec_map.shape
        hidden_states = [cell.init_hidden(B, H, W, tec_map.device) for cell in self.encoder_cells]
        
        for t in range(T):
            input_t = self._build_input(tec_map[:, t], aux_x[:, t], H, W)
            for i, cell in enumerate(self.encoder_cells):
                if i == 0:
                    hidden_states[i] = cell(input_t, hidden_states[i])
                else:
                    hidden_states[i] = cell(hidden_states[i-1][0], hidden_states[i])
        
        return hidden_states

    def _decode_step(self, dec_input, dec_states):
        for i, cell in enumerate(self.decoder_cells):
            h_prev, c_prev = dec_states[i]
            if i == 0:
                dec_states[i] = cell(dec_input, (h_prev, c_prev))
            else:
                dec_states[i] = cell(dec_states[i-1][0], (h_prev, c_prev))
        return self.head(dec_states[-1][0])

    def forward(self, tec_map, aux_x=None, future_aux=None, dec_aux=None,
                return_hidden=False, y_true=None, tf_ratio=None):
        B, _, _, H, W = tec_map.shape

        # 编码历史序列
        hist_states = self._encode_sequence(tec_map, aux_x)

        # 解码（无 FiLM 调制）
        decoder_states = [(h, c) for h, c in hist_states]
        outputs = []
        base = tec_map[:, -1]
        spatial_grid = base[:, 1:] if base.shape[1] > 1 else None

        for step in range(self.pred_steps):
            # 早期融合：直接在输入层拼接辅助特征
            aux_step = dec_aux[:, step] if dec_aux is not None else torch.zeros(B, self.num_aux, device=tec_map.device)
            dec_input = self._build_input(base, aux_step, H, W)
            pred_frame = self._decode_step(dec_input, decoder_states)
            outputs.append(pred_frame)

            use_tf = (self.training and y_true is not None and tf_ratio is not None 
                      and random.random() < tf_ratio)
            next_tec = y_true[:, step] if use_tf else pred_frame
            base = torch.cat([next_tec, spatial_grid], dim=1) if spatial_grid is not None else next_tec

        output = torch.stack(outputs, dim=1)
        
        if return_hidden:
            return output, [state[0] for state in hist_states]
        return output



