from block import *
from util import *
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader


# ============================================================
# ============================ CONFIG ========================
# ============================================================

# ---- Random seed ----
SEED = 128

# ---- Training hyper-parameters ----
MAX_EPOCHS = 500
BATCH_SIZE = 180
SAVE_INTERVAL = 50                    # save checkpoints every N epochs

# ---- Loss weights ----
LAMBDA1 = 1
LAMBDA2 = 4

# ---- Model / data hyper-parameters ----
SPECTRA_LENGTH = 201
INPUT_DIM = 201                       # ConvResidualModule_seperate input_dim
RES_DIM = 201                         # ConvResidualModule_seperate res_dim
REPEAT_NUM = 31                       # repeat_interleave count / reshape factor
TRANS_ENCODER_CHANNELS = [16, 32, 64]
TRANS_ENCODER_STRIDE = 2
INIT_METHOD = "normal"
RECONSTRUCT_ACTIVATION = nn.ReLU()

# ---- Optimizer hyper-parameters ----
OPTIMIZER_RECONSTRUCT = "adam"
OPTIMIZER_TRANS_ENCODER = "adam"
LR_RECONSTRUCT = 0.0001
LR_TRANS_ENCODER = 0.0001

# ---- Device ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Input files ----
TRANS_MATRIX_META1_PATH = "metasurface_1_trans_matrix.pt"
DATASET_PATH = "smooth_multipeak.pt"

# ---- Output paths ----
CHECKPOINT_DIR = "./checkpoints_spectrum"

TRANS_ENCODER_CKPT_FMT = f"{CHECKPOINT_DIR}/transmission_encoder_epoch_{{epoch}}.pth"
RECONSTRUCT_CKPT_FMT = f"{CHECKPOINT_DIR}/resnet_reconstruction_epoch_{{epoch}}.pth"

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


class SyntheticPeaksDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



trans_matrix_meta1 = torch.load(TRANS_MATRIX_META1_PATH, map_location="cpu", weights_only=True).float()


loaded_data = torch.load(DATASET_PATH, map_location="cpu", weights_only=True).float()
dataset_split = torch.load("./checkpoints_angle/dataset_split.pt", map_location="cpu", weights_only=True)
if dataset_split["dataset_sha256"] != spectrum_fingerprint(loaded_data):
    raise ValueError("Spectrum dataset differs from the angle-training dataset")
train_indices = dataset_split["train_indices"]
test_indices = dataset_split["test_indices"]
combined_indices = torch.cat((train_indices, test_indices))
if not len(train_indices) or not len(test_indices) or len(combined_indices) != len(loaded_data) or not torch.equal(combined_indices.sort().values, torch.arange(len(loaded_data))):
    raise ValueError("Training and test indices must partition the spectrum dataset without overlap")
train_dataset = SyntheticPeaksDataset(loaded_data[train_indices])

dataloader_for_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


resnet_reconstruction = ConvResidualModule_seperate(input_dim=INPUT_DIM, res_dim=RES_DIM,
                                                    init_method=INIT_METHOD,
                                                    activation=RECONSTRUCT_ACTIVATION)


transmission_encoder = TransmissionEncoder(spectra_length=SPECTRA_LENGTH, num=trans_matrix_meta1.shape[1],
                                           channels=TRANS_ENCODER_CHANNELS, strid=TRANS_ENCODER_STRIDE,
                                           init_method=INIT_METHOD)


intensity_encoder = IntensityEncoder(input_dim=120, embedding_dim=120, init_method=INIT_METHOD).to(DEVICE)
project_block = Project(dims=[240, 512, 1024, 512, 201], dropout_rate=0, init_method=INIT_METHOD).to(DEVICE)
for module, filename in (
    (transmission_encoder, "transmission_encoder_final.pth"),
    (intensity_encoder, "intensity_encoder_final.pth"),
    (project_block, "project_block_final.pth"),
):
    module.load_state_dict(torch.load(os.path.join("./checkpoints_angle", filename), map_location=DEVICE, weights_only=True))
    module.requires_grad_(False)
    module.eval()

resnet_reconstruction = resnet_reconstruction.to(DEVICE)
trans_matrix_meta1 = trans_matrix_meta1.to(DEVICE)
transmission_encoder = transmission_encoder.to(DEVICE)


meta1_transposed = trans_matrix_meta1.transpose(1, 2)


meta1_normalized = meta1_transposed / meta1_transposed.max()


optimizer_reconstruct = get_optimizer(OPTIMIZER_RECONSTRUCT, resnet_reconstruction.parameters(), LR_RECONSTRUCT)


criterion_mse = nn.MSELoss(reduction='mean')


train_losses = []


for epoch in range(MAX_EPOCHS):

    i = 0
    mse_loss_sum = 0.0
    mse_loss_sum_for_test = 0.0
    l1_loss_sum = 0
    l2_loss_sum = 0

    progress_bar = tqdm(dataloader_for_train, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS}")
    resnet_reconstruction.train()
    transmission_encoder.eval()
    intensity_encoder.eval()
    project_block.eval()

    for batch_idx, batch_data in enumerate(progress_bar):

        i = i + 1

        batch_data = batch_data.to(DEVICE)
        current_batch_size = batch_data.shape[0]


        optimizer_reconstruct.zero_grad()


        with torch.no_grad():
            input_reshaped = batch_data.view(current_batch_size, 1, 1, SPECTRA_LENGTH)

            target1 = torch.matmul(input_reshaped, meta1_normalized)


            final_target = target1.reshape(-1, trans_matrix_meta1.shape[1])


            target_l1 = batch_data
            target_l1 = torch.repeat_interleave(target_l1, repeats=REPEAT_NUM, dim=0)

            combined_results = final_target.data.detach()


        with torch.no_grad():
            output_a = transmission_encoder(meta1_normalized.transpose(1, 2))
            output_a_repeated = output_a.repeat(current_batch_size, 1)
            project_input = torch.cat((intensity_encoder(combined_results), output_a_repeated), dim=1)
            final_input = F.normalize(project_block(project_input), p=2, dim=1)

        reconstruction_output = resnet_reconstruction(final_input)

        l1_loss = criterion_mse(reconstruction_output, target_l1)

        final_output = ((reconstruction_output.reshape(current_batch_size, REPEAT_NUM, 1, SPECTRA_LENGTH)) @ meta1_normalized).reshape(current_batch_size * REPEAT_NUM, trans_matrix_meta1.shape[1])

        l2_loss = criterion_mse(final_output, final_target)

        final_loss = LAMBDA1 * l1_loss + LAMBDA2 * l2_loss


        final_loss.backward()

        optimizer_reconstruct.step()



        mse_loss_train_batch = final_loss.item()
        l1_loss_sum += l1_loss.item()
        l2_loss_sum += l2_loss.item()
        mse_loss_sum += mse_loss_train_batch

        progress_bar.set_postfix({'mean loss': f'{mse_loss_train_batch:.9f}'})


    avg_loss = mse_loss_sum / len(dataloader_for_train)
    avg_l1_loss = l1_loss_sum / len(dataloader_for_train)
    avg_l2_loss = l2_loss_sum / len(dataloader_for_train)

    train_losses.append(avg_loss)

    print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")
    print(f"Average MSE mean Loss: {avg_loss:.9f}")
    print(f"Average l1_Loss: {avg_l1_loss:.9f}")
    print(f"Average l2_loss: {avg_l2_loss:.9f}")


    if (epoch + 1) % SAVE_INTERVAL == 0 or epoch == MAX_EPOCHS - 1:

        torch.save(transmission_encoder.state_dict(),
                   TRANS_ENCODER_CKPT_FMT.format(epoch=epoch + 1))


        torch.save(resnet_reconstruction.state_dict(),
                   RECONSTRUCT_CKPT_FMT.format(epoch=epoch + 1))

        print(f"Checkpoint saved at epoch_{epoch + 1}")


print("Training finished.")
