import torch
import numpy as np

# 计算sim

def sim(z_i, z_j):
    # z_i 和 z_j 均为 1-d tensor
    norm_dot_product = torch.dot(z_i, z_j) / (torch.linalg.norm(z_i) * torch.linalg.norm(z_j))
    return norm_dot_product

def simclr_loss_naive(out_left, out_right, tau):
    """
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples

    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]

    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k + N]

        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 计算 l(k, k+N)
        # 计算左边的分子
        left_numerator = (sim(z_k, z_k_N) / tau).exp()
        # 计算左边的分母中需要进行sim运算的元素
        left_need_sim = out[np.arange(2 * N) != k]
        # 计算左边的分母
        left_denominator = torch.tensor([sim(z_k, z_i) / tau for z_i in left_need_sim]).exp().sum()
        # 计算左边的结果
        left = -(left_numerator / left_denominator).log()

        # 计算 l(k+N, k)
        # 计算右边的分子
        right_numerator = (sim(z_k_N, z_k) / tau).exp()
        # 计算右边的分母中需要进行sim运算的元素
        right_need_sim = out[np.arange(2 * N) != k + N]
        # 计算右边的分母
        right_denominator = torch.tensor([sim(z_k_N, z_i) / tau for z_i in right_need_sim]).exp().sum()
        # 计算右边的结果
        right = -(right_numerator / right_denominator).log()

        total_loss += left + right

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2 * N)
    return total_loss


import torch

def simclr_loss_vectorized(out_left, out_right, tau=0.5):
    """
    Vectorized NT-Xent loss (SimCLR) without any loops.

    Inputs:
    - out_left: [N, D] tensor
    - out_right: [N, D] tensor
    - tau: temperature scalar

    Returns:
    - loss: scalar tensor
    """
    N = out_left.shape[0]
    
    # 1. 拼接左右分支
    out = torch.cat([out_left, out_right], dim=0)  # [2N, D]
    
    # 2. L2 normalize 每个向量
    out = out / out.norm(dim=1, keepdim=True)
    
    # 3. 计算相似度矩阵
    sim_matrix = torch.matmul(out, out.T)  # [2N, 2N], cosine similarity
    
    # 4. 指数化并除以 tau
    sim_matrix = sim_matrix / tau
    sim_matrix_exp = torch.exp(sim_matrix)
    
    # 5. 对角线元素不参与 softmax（排除自身）
    mask = (~torch.eye(2*N, dtype=bool)).to(out.device)  # mask[i,i] = False
    sim_matrix_exp = sim_matrix_exp * mask
    
    # 6. 分子：正样本对
    pos_idx = torch.arange(N)
    numerator = torch.exp(sim_matrix[pos_idx, pos_idx + N] / tau)
    numerator = torch.cat([numerator, torch.exp(sim_matrix[pos_idx + N, pos_idx] / tau)], dim=0)  # [2N]
    
    # 7. 分母：每行所有非自身元素求和
    denominator = sim_matrix_exp.sum(dim=1)  # [2N]
    
    # 8. NT-Xent loss
    loss = -torch.log(numerator / denominator)
    loss = loss.mean()
    
    return loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None

    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 看看公式，sim 是一个除法，但是可以看做两个除法相乘
    norm_left = out_left / torch.linalg.norm(out_left, dim=1, keepdim=True)
    norm_right = out_right / torch.linalg.norm(out_right, dim=1, keepdim=True)

    pos_pairs = torch.sum(norm_left * norm_right, dim=1, keepdim=True)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


if __name__ == "__main__":
    torch.manual_seed(12)  # 保证可重复

    N = 4   # batch size
    D = 128  # 特征维度
    tau = 0.5

    # 随机生成模拟输出向量
    out_left = torch.randn(N, D)
    out_right = torch.randn(N, D)

    # 计算损失
    loss = simclr_loss_naive(out_left, out_right, tau)
    loss2 = simclr_loss_vectorized(out_left, out_right, tau)
    print("SimCLR naive loss:", loss.item())
    print("SimCLR vectorized loss:",loss2.item())