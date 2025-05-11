#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
// #include "module_ispc.h"
// using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX)
{
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX) + y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float val)
{
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b,
                         const int &sizeX, const int &sizeY, const int &sizeZ)
{
    return tensor[x * (sizeX * sizeY * sizeZ) // skipping over full x blocks
                  + y * (sizeY * sizeZ)       // skipping over full y blocks within the x block
                  + z * (sizeZ) + b];         // skipping over full z blocks within (x,y) + offset 'b'
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b,
                         const int &sizeX, const int &sizeY, const int &sizeZ, float val)
{
    // return;
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor)
{
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 *
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                               int B, int H, int N, int d)
{

    // Q, K, V are passed in with Shape: (B, H, N, d)
    // QK^t Intermediate Tensor has Shape (N, N)

    // Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    // Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
           for (int j = 0; j < N; j++) {
               float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
               twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */

    // -------- YOUR CODE HERE  -------- //
    // loop over Batch Size
    for (int b = 0; b < B; b++)
    {
        // loop over Heads
        for (int h = 0; h < H; h++)
        {
            // part a of 2)
            // applying QK^t and storing into NxN temp
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    twoDimWrite(QK_t, i, j, N, 0); // setting all values to zero initially

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < d; k++)
                        sum += fourDimRead(Q, b, h, i, k, H, N, d) * fourDimRead(Q, b, h, j, k, H, N, d); // Q[b][h][i][k] × K[b][h][j][k]
                    // kk = K[b][h][j][k]; ^^^
                    twoDimWrite(QK_t, i, j, N, sum); // writing result back into 2D Vector
                }
            }

            // part b of 2)
            // looping over each row now
            for (int i = 0; i < N; i++)
            {
                /*
                    For each row, you should get the exponential of each row element,
                    which you can get using the C++ inbuilt `exp` function. Now, divide each of these
                    resulting exponentials by the sum of all exponentials in its row and then store it back into QK^t.
                */
                float sum_r = 0;
                for (int j = 0; j < N; j++)
                {
                    float exponent = exp(twoDimRead(QK_t, i, j, N));
                    twoDimWrite(QK_t, i, j, N, exponent); // writing exponent into QK^T
                    sum_r += exponent;
                }

                for (int j = 0; j < N; j++)
                {
                    float exponent = twoDimRead(QK_t, i, j, N);   // reading exponent value
                    twoDimWrite(QK_t, i, j, N, exponent / sum_r); // exponent / sum(exponents)
                }
            }

            // part c of 2)
            // matrix multiply QK^t with V and store the result into O.
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    float val = 0;
                    for (int k = 0; k < N; k++)
                    {
                        val += twoDimRead(QK_t, i, k, N) * fourDimRead(V, b, h, k, j, H, N, d);
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, val);
                }
            }
        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                                        int B, int H, int N, int d)
{

    // Q, K, V are passed in with Shape: (B, H, N, d)
    // QK^t Intermediate Tensor has Shape (N, N)

    // Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    // Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    const int TILE_N = 16; // block size for the sequence dimension (N×N)
    const int TILE_D = 16; // block size for the feature dimension (d)
    for (int b = 0; b < B; b++)
    {
        // loop over Heads
        for (int h = 0; h < H; h++)
        {
            // default values set to 0.00
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    twoDimWrite(QK_t, i, j, N, 0);

            // blocked matrix multiply
            // three-level blocking to efficiently use small chunks in L1 cache
            for (int i = 0; i < N; i += TILE_N)
            {
                int max_i = std::min(N, i + TILE_N); // for handiling 'remainder' tiles
                for (int j = 0; j < N; j += TILE_N)
                {
                    int max_j = std::min(N, j + TILE_N);
                    for (int k = 0; k < d; k += TILE_D)
                    {
                        int max_k = std::min(d, k + TILE_D);
                        // multiplying TILE_N×TILE_D block of Q by the TILE_D×TILE_N block of K^T
                        for (int temp_i = i; temp_i < max_i; temp_i++)
                        {
                            for (int temp_k = k; temp_k < max_k; temp_k++)
                            {
                                float q = fourDimRead(Q, b, h, temp_i, temp_k, H, N, d);
                                for (int temp_j = j; temp_j < max_j; j++)
                                {
                                    float kay = fourDimRead(K, b, h, temp_j, temp_k, H, N, d);
                                    float prev = twoDimRead(QK_t, temp_i, temp_j, N); // accessing val of QK_t
                                    float temp_qk = prev + q * kay;
                                    twoDimWrite(QK_t, temp_i, temp_j, N, temp_qk);
                                }
                            }
                        }
                    }
                }
            }

            // softmax -> same as above in part 2
            for (int i = 0; i < N; i++)
            {
                float sum_r = 0;
                for (int j = 0; j < N; j++)
                {
                    float exponent = expf(twoDimRead(QK_t, i, j, N));
                    twoDimWrite(QK_t, i, j, N, exponent); // writing exponent into QK^T
                    sum_r += exponent;
                }

                for (int j = 0; j < N; j++)
                {
                    float exponent = twoDimRead(QK_t, i, j, N);   // reading exponent value
                    twoDimWrite(QK_t, i, j, N, exponent / sum_r); // exponent / sum(exponents)
                }
            }

            // blocked matrix multiply P (N×N) × V[b,h] ------ todo
            for (int i = 0; i < N; i += TILE_N)
            {
                int max_i = std::min(N, i + TILE_N);

                for (int j = 0; j < d; j += TILE_D)
                {
                    int max_j = std::min(d, j + TILE_D);

                    for (int k = 0; k < N; k += TILE_N)
                    {
                        int max_k = std::min(N, k + TILE_N);

                        // multiplying TILE_N×TILE_D block of Q by the TILE_D×TILE_N block of K^T
                        for (int temp_i = i; temp_i < max_i; temp_i++)
                        {
                            for (int temp_k = k; temp_k < max_k; temp_k++)
                            {
                                float p = twoDimRead(QK_t, temp_i, temp_k, N);
                                for (int temp_j = j; temp_j < max_j; j++)
                                {
                                    float v = fourDimRead(V, b, h, temp_j, temp_k, H, N, d);

                                    // adding into O matrix
                                    float prev = fourDimRead(O, b, h, temp_i, temp_k, H, N, d);
                                    prev += p * v;
                                    fourDimWrite(O, b, h, temp_i, temp_k, H, N, d, prev); // p*v
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                               int B, int H, int N, int d)
{

    // Q, K, V are passed in with Shape: (B, H, N, d)

    // Make O Tensor with Shape (B, H, N, d)
    // and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    // Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // Format ORow Tensor into a 1D vector
    //  You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);

// -------- YOUR CODE HERE  -------- //
// We give you a template of the first three loops for your convenience
// Parallelize over batch, head, and output‐row all at once
#pragma omp parallel for collapse(3)
    // loop over batch
    for (int b = 0; b < B; b++)
    {
        // loop over heads
        for (int h = 0; h < H; h++)
        {
            for (int i = 0; i < N; i++)
            {
                // YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});
                std::vector<float> ORow = formatTensor(ORowTensor);
                // YOUR CODE HERE

                // first compute one row of QK^t
                for (int j = 0; j < N; j++)
                {
                    float temp = 0.0f;

                    for (int k = 0; k < d; k++)
                    {
                        float q = fourDimRead(Q, b, h, i, k, H, N, d);
                        float kval = fourDimRead(K, b, h, j, k, H, N, d);
                        temp += q * kval; // Q[ik]*K[jk]
                    }
                    ORow[j] = temp;
                }

                // softmax current row
                float sum = 0.0f;
                for (int j = 0; j < N; j++)
                {
                    sum += exp(ORow[j]);
                }
                for (int j = 0; j < N; j++)
                {
                    ORow[j] /= sum;
                }

                // multiply the softmax'd row by V to fully compute the first row of our attention output
                for (int j = 0; j < d; j++)
                {
                    float temp = 0.0f;
                    for (int k = 0; k < N; k++)
                    {
                        float v = fourDimRead(V, b, h, k, j, H, N, d);
                        temp += ORow[k] * v;
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, temp);
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
                               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
                               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
                               torch::Tensor OiTensor, torch::Tensor LTensor, torch::Tensor LiTensor,
                               torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                               int B, int H, int N, int d)
{

    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    // Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    // Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    // blockColSize = Bc, blockRowSize = Br
    // lines 2-5 of pseudocode
    int Tc = (N + Bc - 1) / Bc; // Tc = N / Bc , number of column‐blocks
    int Tr = (N + Br - 1) / Br; // Tr = N / Br , number of row‐blocks

    for (int b = 0; b < B; b++)
    {
        for (int h = 0; h < H; h++)
        {
            // inserting zeros in l
            for (int i = 0; i < N; i++)
            {
                l[i] = 0.0f;
            }

            // loop over column blocks
            for (int j = 0; j < Tc; j++)
            {
                int j_Bc = j * Bc;
                int bc = std::min(Bc, N - j_Bc); // actual width of block

                for (int jj = 0; jj < bc; jj++)
                {
                    int globalCol = j_Bc + jj;
                    for (int k = 0; k < d; k++)
                    {
                        float kval = fourDimRead(K, b, h, globalCol, k, H, N, d);
                        float vval = fourDimRead(V, b, h, globalCol, k, H, N, d);

                        twoDimWrite(Kj, jj, k, d, kval);
                        twoDimWrite(Vj, jj, k, d, vval);
                    }
                }

                //  7) Divide Q into Tr blocks Q0…QTr of size (Br × d)
                //  10) Divide l into Tr blocks l0…lTr of size (Br)
                //  11) Divide O into Tr blocks O0…OTr of size (Br × d)
                // loop over row blocks
                for (int i = 0; i < Tr; i++)
                {
                    int i_Br = i * Br;
                    int br = std::min(Br, N - i_Br);

                    // load Q_i into Qi, O_i into Oi, l_i into li
                    for (int ii = 0; ii < br; ii++)
                    {
                        int globalRow = i_Br + ii;
                        for (int index = 0; index < d; index++)
                        {
                            float qval = fourDimRead(Q, b, h, globalRow, index, H, N, d);
                            float oval = fourDimRead(O, b, h, globalRow, index, H, N, d);
                            twoDimWrite(Qi, ii, index, d, qval);
                            twoDimWrite(Oi, ii, index, d, oval);
                        }
                        // copy the running normalizer
                        li[ii] = l[globalRow];
                    }

                    // Compute Sij = QK^t
                    for (int ii = 0; ii < br; ii++)
                    {
                        for (int jj = 0; jj < bc; jj++)
                        {
                            float mul = 0.0f;
                            for (int k = 0; k < d; k++)
                            {
                                mul += twoDimRead(Qi, ii, k, d) * twoDimRead(Kj, jj, k, d);
                            }
                            twoDimWrite(Sij, ii, jj, bc, mul);
                        }
                    }
                    // Compute Pij = exp(Sij)
                    for (int ii = 0; ii < br; ii++)
                    {
                        for (int jj = 0; jj < bc; jj++)
                        {
                            twoDimWrite(Pij, ii, jj, bc, expf(twoDimRead(Sij, ii, jj, bc)));
                        }
                    }
                    // for(int index=0;index<br*bc;index++){
                    //     Pij[index] = exp(Sij[idx]);
                    // }

                    // Compute Lij = rowsum(Pij)
                    for (int ii = 0; ii < br; ii++)
                    {
                        float sum = 0.0f;
                        for (int jj = 0; jj < bc; jj++)
                        {
                            sum += twoDimRead(Pij, ii, jj, bc);
                        }
                        lij[ii] = sum;
                    }

                    // Compute lnew=li + lij
                    for (int ii = 0; ii < br; ii++)
                    {
                        lnew[ii] = li[ii] + lij[ii];
                    }

                    // Compute Oi = (liOi + PijVj) /
                    //   Pij*Vj
                    for (int ii = 0; ii < br; ii++)
                    {
                        for (int k = 0; k < d; k++)
                        {
                            float mul = 0.0f;
                            for (int jj = 0; jj < bc; jj++)
                            {
                                mul += Pij[ii * bc + jj] * Vj[jj * d + k];
                            }
                            PV[ii * d + k] = mul;

                            // //read old scaled Oi
                            // float oldOi = twoDimRead(Oi, ii, k, d);
                            // mul = li[ii] * oldOi + mul;
                            // // divide by the new normalizer
                            // mul /= lnew[ii];
                            // // write back into the local Oi tile
                            // twoDimWrite(Oi, ii, k, d, mul);
                        }
                    }
                    for (int ii = 0; ii < br; ii++)
                    {
                        float inv_lnew = 1.0f / lnew[ii];
                        for (int jj = 0; jj < d; jj++)
                        {
                            float old = twoDimRead(Oi, ii, jj, d); // previous local Oi[ii][jj]
                            float updated = (li[ii] * old + PV[ii * d + jj]) * inv_lnew;
                            twoDimWrite(Oi, ii, jj, d, updated); // update local Oi
                        }
                    }

                    //  liOi -> elementwise multiplication
                    for (int ii = 0; ii < br; ii++)
                    {
                        int globalRow = i_Br + ii;
                        for (int k = 0; k < d; k++)
                        {
                            float newval = twoDimRead(Oi, ii, k, d);
                            // O[b][h][globalRow][k] = newval
                            fourDimWrite(O, b, h, globalRow, k, H, N, d, newval);
                        }
                        // l[globalRow] = lnew[ii]
                        l[globalRow] = lnew[ii];
                    }
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
    m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
    m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
    m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
    m.def("twoDimRead", &twoDimRead, "twoDimRead");
    m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
