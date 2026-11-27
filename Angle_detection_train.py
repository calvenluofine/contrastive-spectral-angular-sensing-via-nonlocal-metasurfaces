from util import *
from block import *
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# ============================ CONFIG ========================
# ============================================================

# ---- Random seed ----
SEED = 128

# ---- Training hyper-parameters ----
MAX_EPOCHS = 500
BATCH_SIZE = 40
TEST_SIZE = 0.2                       # train/test split ratio for train_test_split
SAVE_INTERVAL = 50                    # save checkpoints every N epochs

# ---- Model / data hyper-parameters ----
SPECTRA_LENGTH = 201
GROUP_SIZE = 5                        # InfoNCE group size / mask num / repeat_interleave count
TRANS_ENCODER_CHANNELS = [16, 32, 64]
TRANS_ENCODER_STRIDE = 2
PROJECT_DIMS = [240, 512, 1024, 512, 201]
PROJECT_USE_BN = False
PROJECT_DROPOUT = 0
INIT_METHOD = "normal"

# ---- Loss / optimizer hyper-parameters ----
TEMPERATURE = 0.05
OPTIMIZER_NAME = "adam"
LR_TRANS_ENCODER = 0.0001
LR_PROJECT_BLOCK = 0.0001

# ---- Device ----
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---- Input files ----
TRANS_MATRIX_META1_PATH = "metasurface_1_trans_matrix.pt"
TRANS_MATRIX_META2_PATH = "metasurface_2_trans_matrix.pt"
DATASET_PATH = "smooth_multipeak.pt"

# ---- Output paths ----
CHECKPOINT_DIR = "./checkpoints_angle"

TRANS_ENCODER_CKPT_FMT = f"{CHECKPOINT_DIR}/transmission_encoder_epoch_{{epoch}}.pth"
PROJECT_BLOCK_CKPT_FMT = f"{CHECKPOINT_DIR}/project_block_epoch_{{epoch}}.pth"

FINAL_TRANS_ENCODER_PATH = f"{CHECKPOINT_DIR}/transmission_encoder_final.pth"
FINAL_PROJECT_BLOCK_PATH = f"{CHECKPOINT_DIR}/project_block_final.pth"
FINAL_INTENSITY_ENCODER_PATH = f"{CHECKPOINT_DIR}/intensity_encoder_final.pth"

# ============================================================
# ========================== CONFIG END ======================
# ============================================================


import os
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def info_nce_loss_vectorized(features, mask, group_size=GROUP_SIZE, temperature=TEMPERATURE):

    n = features.shape[0] // 2
    p1, p2 = features[:n], features[n:]

    p1_norm = F.normalize(p1, p=2, dim=1)
    p2_norm = F.normalize(p2, p=2, dim=1)

    similarity = torch.matmul(p1_norm, p2_norm.T) / temperature

    num_groups = n // group_size
    grouped_similarity = similarity.view(num_groups, group_size, -1)
    grouped_mask = mask.view(num_groups, group_size, -1)

    exp_sim = torch.exp(grouped_similarity)
    group_denominator = exp_sim.sum(dim=(1, 2))
    log_denominator = torch.log(group_denominator)

    masked_exp = exp_sim * grouped_mask
    group_numerator = masked_exp.sum(dim=(1, 2))
    log_numerator = torch.log(group_numerator)

    log_probs = log_numerator - log_denominator

    loss = -log_probs.mean()

    return loss


class SyntheticPeaksDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


trans_matrix_meta1 = torch.load(TRANS_MATRIX_META1_PATH, map_location="cpu", weights_only=True).float()
trans_matrix_meta2 = torch.load(TRANS_MATRIX_META2_PATH, map_location="cpu", weights_only=True).float()


print("=== Loading and splitting dataset ===")
loaded_data = torch.load(DATASET_PATH, map_location="cpu", weights_only=True).float()
print(f"Raw data shape: {loaded_data.shape}")


data_numpy = loaded_data.numpy()


train_indices, test_indices = train_test_split(np.arange(len(data_numpy)), test_size=TEST_SIZE, random_state=SEED, shuffle=True)
train_data, test_data = data_numpy[train_indices], data_numpy[test_indices]
torch.save({
    "train_indices": torch.as_tensor(train_indices, dtype=torch.long),
    "test_indices": torch.as_tensor(test_indices, dtype=torch.long),
    "dataset_sha256": spectrum_fingerprint(loaded_data),
}, os.path.join(CHECKPOINT_DIR, "dataset_split.pt"))


train_data = torch.from_numpy(train_data).float()
test_data = torch.from_numpy(test_data).float()

print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")


train_dataset = SyntheticPeaksDataset(train_data)
test_dataset = SyntheticPeaksDataset(test_data)


dataloader_for_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dataloader_for_test = DataLoader(test_dataset, batch_size=min(BATCH_SIZE, 4), shuffle=False)

print(f"Train batches: {len(dataloader_for_train)}")
print(f"Test batches: {len(dataloader_for_test)}")


transmission_encoder = TransmissionEncoder(spectra_length=SPECTRA_LENGTH, num=trans_matrix_meta1.shape[1],
                                           channels=TRANS_ENCODER_CHANNELS, strid=TRANS_ENCODER_STRIDE,
                                           init_method=INIT_METHOD)

intensity_encoder = IntensityEncoder(input_dim=120, embedding_dim=120, init_method=INIT_METHOD)

project_block = Project(dims=PROJECT_DIMS, use_batch_norm=PROJECT_USE_BN, dropout_rate=PROJECT_DROPOUT,
                        init_method=INIT_METHOD)


transmission_encoder = transmission_encoder.to(DEVICE)
intensity_encoder = intensity_encoder.to(DEVICE)
project_block = project_block.to(DEVICE)
trans_matrix_meta1 = trans_matrix_meta1.to(DEVICE)
trans_matrix_meta2 = trans_matrix_meta2.to(DEVICE)


meta1_normalized = trans_matrix_meta1 / trans_matrix_meta1.max()
meta2_normalized = trans_matrix_meta2 / trans_matrix_meta2.max()


optimizer_1 = get_optimizer(OPTIMIZER_NAME, list(transmission_encoder.parameters()) + list(intensity_encoder.parameters()), LR_TRANS_ENCODER)
optimizer_2 = get_optimizer(OPTIMIZER_NAME, project_block.parameters(), LR_PROJECT_BLOCK)


if GROUP_SIZE > meta1_normalized.shape[0] or meta1_normalized.shape != meta2_normalized.shape:
    raise ValueError("Transmission matrices must have matching shapes and enough calibrated angles")


train_losses = []
test_losses = []

for epoch in range(MAX_EPOCHS):
    total_loss_sum = 0.0
    simclr_loss_sum = 0.0
    mse_loss_sum = 0.0

    i = 0


    progress_bar = tqdm(dataloader_for_train, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS}")


    transmission_encoder.train()
    intensity_encoder.train()
    project_block.train()

    for batch_idx, batch_data in enumerate(progress_bar):

        batch_data = batch_data.to(DEVICE)
        current_batch_size = batch_data.shape[0]
        selected_angles = torch.randperm(meta1_normalized.shape[0], device=DEVICE)[:GROUP_SIZE]
        simclr_mask = generate_mask_matrix(current_batch_size, DEVICE, num=GROUP_SIZE)

        i = i + 1

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()


        with torch.no_grad():
            meta1_transposed = meta1_normalized[selected_angles].transpose(1, 2)
            meta2_transposed = meta2_normalized[selected_angles].transpose(1, 2)


            batch_data = batch_data.view(current_batch_size, 1, 1, SPECTRA_LENGTH)


            result1 = torch.matmul(batch_data, meta1_transposed)
            result2 = torch.matmul(batch_data, meta2_transposed)


            result1_reshaped = result1.reshape(-1, 120)
            result2_reshaped = result2.reshape(-1, 120)

            result1_reshaped_repeat = result1_reshaped.repeat_interleave(GROUP_SIZE, dim=0)
            result2_reshaped_repeat = result2_reshaped.repeat_interleave(GROUP_SIZE, dim=0)


            combined_results = torch.cat([result1_reshaped_repeat, result2_reshaped_repeat],
                                         dim=0)


        output_a = transmission_encoder(meta1_normalized[selected_angles])
        output_b = transmission_encoder(meta2_normalized[selected_angles])


        output_a_repeated = output_a.repeat(current_batch_size * GROUP_SIZE, 1)
        output_b_repeated = output_b.repeat(current_batch_size * GROUP_SIZE, 1)


        encoder_output = torch.cat([output_a_repeated, output_b_repeated], dim=0)


        project_input = torch.cat((intensity_encoder(combined_results), encoder_output), dim=1)


        project_output = project_block(project_input)


        simclr_loss = info_nce_loss_vectorized(project_output, simclr_mask, temperature=TEMPERATURE)


        simclr_loss.backward()

        optimizer_1.step()
        optimizer_2.step()


        simclr_loss_val = simclr_loss.item()

        simclr_loss_sum += simclr_loss_val


        progress_bar.set_postfix({
            'simclr_loss': f'{simclr_loss_val:.4f}',
        })


    print("=== Test set evaluation ===")
    transmission_encoder.eval()
    intensity_encoder.eval()
    project_block.eval()

    test_loss_sum = 0.0
    test_batches = 0

    with torch.no_grad():
        for batch_for_test in dataloader_for_test:
            batch_for_test = batch_for_test.to(DEVICE)
            test_batch_size = batch_for_test.shape[0]
            test_group_size = meta1_normalized.shape[0]
            selected_angles = torch.arange(test_group_size, device=DEVICE)


            test_mask = generate_mask_matrix(test_batch_size, DEVICE, num=test_group_size)

            meta1_transposed = meta1_normalized[selected_angles].transpose(1, 2)
            meta2_transposed = meta2_normalized[selected_angles].transpose(1, 2)

            batch_data_for_test = batch_for_test.view(test_batch_size, 1, 1, SPECTRA_LENGTH)

            result1 = torch.matmul(batch_data_for_test, meta1_transposed)
            result2 = torch.matmul(batch_data_for_test, meta2_transposed)

            result1_reshaped = result1.reshape(-1, 120)
            result2_reshaped = result2.reshape(-1, 120)

            result1_reshaped_repeat = result1_reshaped.repeat_interleave(test_group_size, dim=0)
            result2_reshaped_repeat = result2_reshaped.repeat_interleave(test_group_size, dim=0)

            combined_results = torch.cat([result1_reshaped_repeat, result2_reshaped_repeat], dim=0)

            output_a = transmission_encoder(meta1_normalized[selected_angles])
            output_b = transmission_encoder(meta2_normalized[selected_angles])

            output_a_repeated = output_a.repeat(test_batch_size * test_group_size, 1)
            output_b_repeated = output_b.repeat(test_batch_size * test_group_size, 1)

            encoder_output = torch.cat([output_a_repeated, output_b_repeated], dim=0)

            project_input = torch.cat((intensity_encoder(combined_results), encoder_output), dim=1)

            project_output = project_block(project_input)


            simclr_loss_for_test = info_nce_loss_vectorized(project_output, test_mask, group_size=test_group_size, temperature=TEMPERATURE)

            test_loss_sum += simclr_loss_for_test.item()
            test_batches += 1


    avg_simclr_loss = simclr_loss_sum / len(dataloader_for_train)
    avg_test_loss = test_loss_sum / test_batches


    train_losses.append(avg_simclr_loss)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")
    print(f"Average Training SimCLR Loss: {avg_simclr_loss:.4f}")
    print(f"Average Test SimCLR Loss: {avg_test_loss:.4f}")


    if (epoch + 1) % SAVE_INTERVAL == 0 or epoch == MAX_EPOCHS - 1:

        torch.save(transmission_encoder.state_dict(),
                   TRANS_ENCODER_CKPT_FMT.format(epoch=epoch + 1))


        torch.save(project_block.state_dict(),
                   PROJECT_BLOCK_CKPT_FMT.format(epoch=epoch + 1))

        torch.save(intensity_encoder.state_dict(),
                   os.path.join(CHECKPOINT_DIR, f"intensity_encoder_epoch_{epoch + 1}.pth"))

        print(f"Checkpoint saved at epoch_{epoch + 1}")


torch.save(transmission_encoder.state_dict(), FINAL_TRANS_ENCODER_PATH)
torch.save(project_block.state_dict(), FINAL_PROJECT_BLOCK_PATH)
torch.save(intensity_encoder.state_dict(), FINAL_INTENSITY_ENCODER_PATH)
print(f"Final models saved to '{FINAL_TRANS_ENCODER_PATH}' and '{FINAL_PROJECT_BLOCK_PATH}'")
