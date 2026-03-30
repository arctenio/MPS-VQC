import os
import datetime
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from mindquantum import Circuit, RY, RZ, CNOT, Hamiltonian, QubitOperator, Simulator, ParameterResolver

######################################################################
#                           全局配置                                  #
######################################################################
# 选择量子测量方案：1 → 单比特期望值; 2 → 多比特期望值
MEASUREMENT_SCHEME = 2
EPOCHS = 30
TEST_SIZE_RATE = 0.3
BATCH_SIZE = 128
N_CIRC_LAYERS = 2

# 需要测试的 Qubit 数量列表
QUBIT_COUNTS_TO_TEST = [6] # [4,6,8, 12]

DATASETS_CONFIG = {
    "Dim128_Activated_chi32": "./data_examples/20260104_233741_fused_features_activated_dim128_chi32.npz",
    "Dim128_Simple_chi32":    "./data_examples/20260104_232928_fused_features_simple_dim128_chi32.npz",
}


######################################################################
#                         量子电路辅助函数                             #
######################################################################
def get_mps_circuit(num_qubits, n_layers):
    circuit = Circuit()

    def add_block(circ, qubits, idx):
        q1, q2 = qubits
        circ += RZ(f'alpha_{idx}_L1').on(q1)
        circ += RY(f'beta_{idx}_L1').on(q1)
        circ += RZ(f'gamma_{idx}_L1').on(q1)
        circ += RZ(f'alpha_{idx}_L2').on(q2)
        circ += RY(f'beta_{idx}_L2').on(q2)
        circ += RZ(f'gamma_{idx}_L2').on(q2)
        circ += CNOT.on(q1, q2)
        circ += RZ(f'alpha_{idx}_R1').on(q1)
        circ += RY(f'beta_{idx}_R1').on(q1)
        circ += RZ(f'gamma_{idx}_R1').on(q1)
        circ += RZ(f'alpha_{idx}_R2').on(q2)
        circ += RY(f'beta_{idx}_R2').on(q2)
        circ += RZ(f'gamma_{idx}_R2').on(q2)

    for layer in range(n_layers):
        for i in range(num_qubits - 2):
            add_block(circuit, [i, i + 1], i + layer * n_layers)
        circuit += RZ(f'alpha_z1_last').on(num_qubits - 1)
        circuit += RY(f'beta_y_last').on(num_qubits - 1)
        circuit += RZ(f'gamma_z2_last').on(num_qubits - 1)
    return circuit


def get_encoder_circuit(num_qubits):
    encoder = Circuit()
    for i in range(num_qubits):
        encoder += RY(f"x{i}").on(i)
    return encoder


######################################################################
#                      量子层定义                                     #
######################################################################
class TorchMQLayer(nn.Module):
    def __init__(self, circuit, observables, n_qubits):
        super().__init__()
        self.sim = Simulator('mqvector', n_qubits)
        self.circuit = circuit
        self.observables = observables
        self.n_qubits = n_qubits
        self.encoder_param_names = [p for p in self.circuit.params_name if p.startswith("x")]
        self.trainable_param_names = [p for p in self.circuit.params_name if not p.startswith("x")]

        self.theta = nn.Parameter(
            torch.empty(len(self.trainable_param_names)).uniform_(-np.pi / 4, np.pi / 4)
        )

    def forward(self, x_batch):
        x_np = x_batch.detach().cpu().numpy()
        theta_np = self.theta.detach().cpu().numpy()
        batch_outputs = []

        # 注意：如果在循环中使用 Simulator，大量创建可能会慢。
        # MindQuantum 的并行处理通常更好，但在 Torch 层中为了简单逻辑保持逐个处理
        # 或者使用 batch processing 接口（如果版本支持）

        for x in x_np:
            self.sim.reset()
            # 确保参数字典匹配
            # x 的维度必须与 encoder 参数数量一致
            encoder_params = ParameterResolver({f"x{i}": float(x[i]) for i in range(len(x))})
            trainable_params = ParameterResolver(
                {name: float(theta_np[idx]) for idx, name in enumerate(self.trainable_param_names)})

            params = ParameterResolver()
            params.update(encoder_params)
            params.update(trainable_params)

            res_list = []
            for obs in self.observables:
                res_val = self.sim.get_expectation(obs, circ_right=self.circuit, pr=params)
                res_list.append(res_val.real)
            batch_outputs.append(res_list)

        return torch.tensor(batch_outputs, dtype=torch.float32, device=x_batch.device)


######################################################################
#                       通用模型类 (动态化)                           #
######################################################################
class HybridPreQuantumClassifier(nn.Module):
    def __init__(self, input_dim, n_qubits, circuit, observables):
        super().__init__()
        # 1. 经典预处理层：将输入维度映射到 n_qubits
        self.fc_embed = nn.Sequential(
            nn.Linear(input_dim, n_qubits),
            nn.LayerNorm(n_qubits),
            nn.Tanh()
        )
        # 2. 量子层
        self.q_layer = TorchMQLayer(circuit, observables, n_qubits)
        # 3. 输出层
        self.fc_out = nn.Linear(len(observables), 1)

    def forward(self, x):
        x_embed = self.fc_embed(x)
        x_embed = (x_embed + 1) * (np.pi / 2)  # 映射到 [0, π]
        q_out = self.q_layer(x_embed)
        return self.fc_out(q_out)


######################################################################
#                       核心实验运行函数                              #
######################################################################
def run_experiment(dataset_name, dataset_path, num_qubits, device):
    print(f"\n{'=' * 60}")
    print(f"Running Experiment: Dataset={dataset_name}, Qubits={num_qubits}")
    print(f"{'=' * 60}")

    # 1. 加载数据
    if not os.path.exists(dataset_path):
        print(f"Error: File not found {dataset_path}")
        return

    data = np.load(dataset_path)
    embeddings = data['features']
    labels = data['labels']

    # 归一化处理
    minmax = MinMaxScaler(feature_range=(0, np.pi))
    X = minmax.fit_transform(embeddings)
    y = labels.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE_RATE, random_state=42, stratify=y
    )

    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # 2. 构建电路和观测算符 (基于当前的 num_qubits)
    encoder = get_encoder_circuit(num_qubits)
    ansatz = get_mps_circuit(num_qubits, n_layers=N_CIRC_LAYERS)
    total_circuit = encoder + ansatz

    if MEASUREMENT_SCHEME == 1:
        observables = [Hamiltonian(QubitOperator('Z0'))]
    elif MEASUREMENT_SCHEME == 2:
        observables = [Hamiltonian(QubitOperator(f'Z{i}')) for i in range(num_qubits)]

    # 3. 初始化模型
    input_dim = embeddings.shape[1]
    model = HybridPreQuantumClassifier(
        input_dim=input_dim,
        n_qubits=num_qubits,
        circuit=total_circuit,
        observables=observables
    ).to(device)

    # 4. 优化器和调度器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    # 使用 CosineAnnealingWarmRestarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=3e-5
    )

    # 5. 设置独立日志文件
    log_dir = "mindquantum_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # 日志文件名包含数据集和qubit数
    log_filename = f"log_{dataset_name}_q{num_qubits}_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)

    with open(log_path, "w") as f:
        f.write(
            f"Experiment Log\nDataset: {dataset_name}\nPath: {dataset_path}\nQubits: {num_qubits}\nEpochs: {EPOCHS}\n\n")

    # 6. 训练循环
    start_time = datetime.datetime.now()

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.unsqueeze(1))
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds.view(-1) == yb.view(-1)).sum().item()
            train_loss += loss.item() * xb.size(0)
            train_total += xb.size(0)

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / train_total

        # 测试
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb.unsqueeze(1))
                preds = (torch.sigmoid(logits) > 0.5).float()
                test_correct += (preds.view(-1) == yb.view(-1)).sum().item()
                test_loss += loss.item() * xb.size(0)
                test_total += xb.size(0)

        test_acc = 100.0 * test_correct / test_total
        avg_test_loss = test_loss / test_total

        # 更新学习率 (WarmRestarts 依赖 epoch 进度，不需要传入 metrics)
        scheduler.step(epoch + epoch / len(train_loader))
        current_lr = scheduler.get_last_lr()[0]

        log_line = (f"Epoch {epoch + 1:03d}/{EPOCHS} | "
                    f"Train Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}% | "
                    f"Test Loss={avg_test_loss:.4f}, Acc={test_acc:.2f}% | "
                    f"LR={current_lr:.6f}\n")

        print(log_line.strip())
        with open(log_path, "a") as f:
            f.write(log_line)

    total_time = datetime.datetime.now() - start_time
    with open(log_path, "a") as f:
        f.write(f"Total Time : {total_time}\n")
    print(f"✅ Finished {dataset_name} with {num_qubits} qubits. Log: {log_path}")

    # 清理显存 (如果是 GPU)
    if device.type == 'cuda':
        torch.cuda.empty_cache()


######################################################################
#                           主程序入口                                #
######################################################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 遍历每个数据集
    for name, path in DATASETS_CONFIG.items():
        # 2. 遍历每个 Qubit 数量配置
        for n_q in QUBIT_COUNTS_TO_TEST:
            try:
                run_experiment(name, path, n_q, device)
            except Exception as e:
                print(f"❌ Error running {name} with {n_q} qubits: {str(e)}")
                # 将错误写入一个单独的 error log
                with open("error_log.txt", "a") as ef:
                    ef.write(f"{datetime.datetime.now()} - {name} - Q{n_q} - {str(e)}\n")