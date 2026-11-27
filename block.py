import torch
import torch.nn as nn
import torch.nn.functional as F


class TransmissionEncoder(nn.Module):
    def __init__(
        self,
        spectra_length=201,
        num=120,
        dropout_rate=0.2,
        use_batch_norm=False,
        channels=[16, 32, 64],
        strid=2,
        init_method="normal",
        init_gain=1.0,
    ):
        super().__init__()
        self.use_dropout = dropout_rate > 0
        self.use_batch_norm = use_batch_norm
        self.strid = strid
        self.init_method = init_method
        self.init_gain = init_gain
        self.num = num

        l1 = (spectra_length - 3) // self.strid + 1
        l2 = (l1 - 3) // self.strid + 1
        l3 = (l2 - 3) // self.strid + 1

        self.conv1 = nn.Conv1d(1, channels[0], kernel_size=3, stride=self.strid)
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size=3, stride=self.strid)
        self.conv3 = nn.Conv1d(channels[1], channels[2], kernel_size=3, stride=self.strid)

        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(channels[0])
            self.bn2 = nn.BatchNorm1d(channels[1])
            self.bn3 = nn.BatchNorm1d(channels[2])

        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Sequential(
            nn.Linear(channels[2] * l3, 512),
            nn.ReLU(),
            nn.Linear(512, num),
            nn.ReLU(),
        )
        self.initialize_weights(init_method, init_gain)

    def initialize_weights(self, method="normal", gain=1.0):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif method == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif method == "uniform":
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                elif method == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=gain)
                else:
                    raise ValueError(f"Unsupported init method: {method}")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif method == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif method == "uniform":
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                elif method == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size, num, spectra_length = x.shape
        assert num == self.num, f"input num {num} does not match expected {self.num}"

        x_reshaped = x.reshape(-1, 1, spectra_length)
        h = self.conv1(x_reshaped)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = self.conv3(h)
        h = F.relu(h)
        h = h.view(batch_size * num, -1)
        h = self.fc(h)
        h_3d = h.view(batch_size, num, num)
        summed = torch.sum(h_3d, dim=1)
        normalized = summed / num
        return normalized


class Project(nn.Module):
    def __init__(
        self,
        dims,
        use_batch_norm=False,
        dropout_rate=0.1,
        activation=nn.ReLU(),
        init_method="kaiming",
        init_gain=1.0,
    ):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output dimensions")

        self.dims = dims
        self.use_batch_norm = use_batch_norm
        self.use_dropout = dropout_rate > 0
        self.activation = activation
        self.init_method = init_method
        self.init_gain = init_gain

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation)

        self.network = nn.Sequential(*layers)
        self.initialize_weights(init_method, init_gain)

    def initialize_weights(self, method="kaiming", gain=1.0):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "kaiming":
                    if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    else:
                        nn.init.kaiming_uniform_(m.weight, a=gain)
                elif method == "xavier":
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif method == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif method == "uniform":
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                elif method == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=gain)
                else:
                    raise ValueError(f"Unsupported init method: {method}")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.size(-1) != self.dims[0]:
            raise ValueError(f"input feature dim {x.size(-1)} does not match expected {self.dims[0]}")
        return self.network(x)


class ConvResidualModule(nn.Module):
    def __init__(
        self,
        input_dim,
        use_dropout=True,
        dropout_rate=0.2,
        use_batch_norm=False,
        activation=nn.ReLU(),
        init_method="kaiming",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        self.activation = activation

        l1 = (input_dim - 3) // 2 + 1
        l2 = (l1 - 3) // 2 + 1
        l3 = (l2 - 3) // 2 + 1
        flattened_size = 64 * l3

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2)

        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(16)
            self.bn2 = nn.BatchNorm1d(32)
            self.bn3 = nn.BatchNorm1d(64)

        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, input_dim)
        self.initialize_weights(init_method)

    def initialize_weights(self, method="kaiming"):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif method == "normal":
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc3(x)
        x = x + residual
        return x


class ConvResidualModule_seperate(nn.Module):
    def __init__(
        self,
        input_dim,
        res_dim,
        use_batch_norm=False,
        activation=nn.ReLU(),
        init_method="kaiming",
    ):
        super().__init__()
        self.res_dim = res_dim
        self.input_dim = input_dim
        self.use_batch_norm = use_batch_norm
        self.activation = activation

        if input_dim != res_dim:
            raise ValueError("Residual decoder input and spectrum dimensions must match")
        self.residual1 = nn.Sequential(nn.Linear(input_dim, 512), activation, nn.Linear(512, res_dim), activation)
        self.residual2 = nn.Sequential(nn.Linear(res_dim, 512), activation, nn.Linear(512, res_dim), activation)
        self.output = nn.Sequential(nn.Linear(res_dim, 512), activation, nn.Linear(512, res_dim))
        self.initialize_weights(init_method)

    def initialize_weights(self, method="kaiming"):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif method == "normal":
                    nn.init.normal_(m.weight, 0, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        return self.output(x)


class IntensityEncoder(nn.Module):
    def __init__(self, input_dim=120, embedding_dim=120, init_method="normal"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
        )
        length = input_dim
        for _ in range(3):
            length = (length - 3) // 2 + 1
        self.fc = nn.Sequential(
            nn.Linear(64 * length, 512), nn.ReLU(), nn.Linear(512, embedding_dim)
        )
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                if init_method == "normal":
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, intensity):
        return self.fc(self.conv(intensity.unsqueeze(1)).flatten(1))
